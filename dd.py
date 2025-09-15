import time
import os
import numpy as np
import cv2
from ultralytics import YOLO
import pygame

# ==================== CONFIG ====================
# --- Camera selection ---
USE_IP_CAMERA = True          # set True to use Hikvision RTSP, False for USB cam
USB_FALLBACK_INDEX = 0        # used if RTSP fails

# --- Hikvision settings ---
HIK_IP        = "169.254.12.77"   # your camera IP
HIK_USER      = "admin"           # <-- set your username
HIK_PASS      = "qwerty@123"      # <-- set your password
HIK_PORT      = 554               # default RTSP port
HIK_CHANNEL   = 102               # 101=main stream (HD), 102=sub stream (lower latency)
RTSP_TRANSPORT = "tcp"            # "tcp" is reliable; "udp" can be lower-latency on LAN

# --- Load shedding / performance ---
RUN_YOLO_EVERY_N = 2        # run YOLO every Nth frame (2 = half the load)
DECODE_DOWNSCALE = 0.75     # resize frame for inference (0.5 or 0.75 are good)
YOLO_DEVICE      = "cpu"    # "cpu" or "cuda"
USE_HANDS        = False    # big CPU saver; set True if you need hands

# --- Detection settings ---
CONF_THRES = 0.35
IOU_THRES  = 0.45
IMG_SIZE   = 448
MODEL_WEIGHTS = "yolov8n-seg.pt"

# Zone polygon (existing feature)
ZONE_POLY_NORM = np.array([
    [0.10, 0.20],  # top-left
    [0.85, 0.20],  # top-right
    [0.85, 0.90],  # bottom-right
    [0.10, 0.90],  # bottom-left
], dtype=np.float32)

# ==== TRIPWIRE ADDITIONS ====
TRIP_LINE_NORM = np.array([[0.02, 0.10], [0.98, 0.10]], dtype=np.float32)  # near top
TRIP_LINE_THICK_PX = 6
TRIP_CUT_PULSE_S   = 2.0
TRIP_DEBOUNCE_S    = 5.0
GPIO_PIN           = 17

MASK_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
ALERT_MP3 = "siren-alert-96052.mp3"

pygame_inited = False
siren_loaded  = False
siren_on      = False

# ==================== RTSP helpers ====================
def set_ffmpeg_env(transport):
    # Low-latency + timeouts; OpenCV uses this only with CAP_FFMPEG
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport};"
        "reorder_queue_size;0;"
        "buffer_size;300000;"
        "max_delay;3000000;"     # 3s
        "stimeout;8000000;"      # 8s (microseconds)
        "flags;low_delay"
    )

def build_rtsp_url(ip, user, pwd, port, channel):
    return f"rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{channel}"

RTSP_URL = build_rtsp_url(HIK_IP, HIK_USER, HIK_PASS, HIK_PORT, HIK_CHANNEL)
set_ffmpeg_env(RTSP_TRANSPORT)

# ==== TRIPWIRE ADDITIONS: power control state ====
gpio_ok = False
relay_on = False
relay_off_at = 0.0
last_trip_time = 0.0
try:
    import RPi.GPIO as GPIO  # type: ignore
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    gpio_ok = True
except Exception:
    gpio_ok = False

def power_relay_on():
    global relay_on, relay_off_at
    if gpio_ok:
        GPIO.output(GPIO_PIN, GPIO.HIGH)
    print("[POWER] CUT-OFF RELAY: ON")
    relay_on = True
    relay_off_at = time.time() + TRIP_CUT_PULSE_S

def power_relay_off():
    global relay_on
    if gpio_ok:
        GPIO.output(GPIO_PIN, GPIO.LOW)
    if relay_on:
        print("[POWER] CUT-OFF RELAY: OFF")
    relay_on = False

def maybe_pulse_power_cut():
    global last_trip_time
    now = time.time()
    if now - last_trip_time >= TRIP_DEBOUNCE_S:
        last_trip_time = now
        power_relay_on()

def _init_audio():
    global pygame_inited, siren_loaded
    if not pygame_inited:
        try:
            pygame.mixer.init()
            pygame_inited = True
        except Exception as e:
            print(f"[AUDIO] mixer init failed: {e}")
            return
    if not siren_loaded:
        try:
            pygame.mixer.music.load(ALERT_MP3)
            siren_loaded = True
        except Exception as e:
            print(f"[AUDIO] failed to load '{ALERT_MP3}': {e}")

def play_alert():
    global siren_on
    if not pygame_inited or not siren_loaded:
        _init_audio()
    if not siren_loaded:
        return
    if not siren_on:
        pygame.mixer.music.play(loops=-1)
        siren_on = True

def stop_alert():
    global siren_on
    if siren_on and pygame_inited:
        pygame.mixer.music.stop()
        siren_on = False

def draw_zone(img, poly, color=(0,0,255)):
    cv2.polylines(img, [poly], True, color, 2)

def draw_trip_line(img, p1, p2, color=(0,255,0), thickness=3):
    cv2.line(img, p1, p2, color, thickness)

def make_line_mask(shape_hw, p1, p2, thickness):
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.line(m, p1, p2, 1, thickness=thickness)
    return m

def line_overlaps_mask(bin_mask, p1, p2, thickness):
    line_m = make_line_mask(bin_mask.shape[:2], p1, p2, thickness)
    return (bin_mask.astype(bool) & line_m.astype(bool)).any()

def mask_overlaps_zone(bin_mask, zone_poly):
    h, w = bin_mask.shape[:2]
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [zone_poly], 1)
    return (bin_mask.astype(bool) & zone_mask.astype(bool)).any()

def point_in_polygon(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def norm_to_pixel(poly_norm, W, H):
    p = poly_norm.copy()
    p[:, 0] = np.clip(p[:, 0] * W, 0, W - 1)
    p[:, 1] = np.clip(p[:, 1] * H, 0, H - 1)
    return p.astype(np.int32)

def norm_line_to_pixel(line_norm, W, H):
    p = norm_to_pixel(line_norm, W, H)
    return (tuple(p[0]), tuple(p[1]))

# ---------- Optional MediaPipe Hands ----------
mp_hands = None
hands = None
if USE_HANDS:
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.25
        )
    except Exception as e:
        print(f"[HANDS] Failed to init MediaPipe Hands: {e}. Continuing without hands.")
        USE_HANDS = False

# ---------- Camera helpers ----------
def open_capture():
    """
    Try opening RTSP first (if enabled), then fall back to USB.
    Returns: (cap, is_rtsp)
    """
    if USE_IP_CAMERA:
        print(f"[VIDEO] Opening Hikvision RTSP: {RTSP_URL} (transport={RTSP_TRANSPORT})")
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency
            except Exception:
                pass
            return cap, True
        else:
            print("[VIDEO] RTSP failed to open, falling back to USB index", USB_FALLBACK_INDEX)

    # USB fallback
    cap = cv2.VideoCapture(USB_FALLBACK_INDEX, cv2.CAP_DSHOW if isinstance(USB_FALLBACK_INDEX, int) else 0)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap, False

def ensure_open(cap, is_rtsp, retry_delay=1.5, max_retries=4):
    """
    Reopen the capture if it has dropped.
    """
    tries = 0
    while not cap.isOpened() and tries < max_retries:
        tries += 1
        print(f"[VIDEO] Reopening stream (attempt {tries}/{max_retries})...")
        cap.release()
        time.sleep(retry_delay)
        cap, is_rtsp = open_capture()
        if cap.isOpened():
            break
    return cap, is_rtsp

def flip_transport_and_reopen(cap):
    """
    If no frames, flip tcp<->udp and try again.
    """
    global RTSP_TRANSPORT
    if not USE_IP_CAMERA:
        return cap, False
    RTSP_TRANSPORT = "udp" if RTSP_TRANSPORT == "tcp" else "tcp"
    print(f"[VIDEO] No frames. Switching RTSP transport to: {RTSP_TRANSPORT}")
    set_ffmpeg_env(RTSP_TRANSPORT)
    cap.release()
    return open_capture()

# ---------- Open camera ----------
cap, is_rtsp = open_capture()
cap, is_rtsp = ensure_open(cap, is_rtsp)
if not cap.isOpened():
    raise RuntimeError("Cannot open any video source (RTSP/USB).")

model = YOLO(MODEL_WEIGHTS)

win_name = "Boundary Intrusion (Seg + Hands)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

last_t = time.time()
frame_idx = 0

while True:
    ok, frame = cap.read()

    # If RTSP briefly drops, try to reconnect and flip transport if needed
    if not ok:
        print("[VIDEO] Frame grab failed; attempting to reconnect...")
        cap, is_rtsp = ensure_open(cap, is_rtsp)
        ok2, frame2 = cap.read()
        if not ok2:
            cap, is_rtsp = flip_transport_and_reopen(cap)
            cap, is_rtsp = ensure_open(cap, is_rtsp)
            ok3, frame3 = cap.read()
            if not ok3:
                print("[VIDEO] Still no frame after reconnect; sleeping briefly.")
                time.sleep(0.5)
                continue
            frame = frame3
        else:
            frame = frame2

    H, W = frame.shape[:2]
    ZONE_POLY = norm_to_pixel(ZONE_POLY_NORM, W, H)
    draw_zone(frame, ZONE_POLY)

    # Draw trip line
    P1, P2 = norm_line_to_pixel(TRIP_LINE_NORM, W, H)
    draw_trip_line(frame, P1, P2, color=(0,255,0), thickness=2)

    # Downscale for inference to reduce load
    if DECODE_DOWNSCALE != 1.0:
        frame_for_infer = cv2.resize(frame, (int(W*DECODE_DOWNSCALE), int(H*DECODE_DOWNSCALE)))
    else:
        frame_for_infer = frame

    run_yolo_now = (frame_idx % RUN_YOLO_EVERY_N == 0)
    any_inside_zone = False
    line_crossed = False

    if run_yolo_now:
        results = model.predict(
            frame_for_infer,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=YOLO_DEVICE,  # "cpu" or "cuda"
            verbose=False,
            agnostic_nms=False,
            max_det=20
        )
        r = results[0]

        # scale back to original frame size for overlays and geometry checks
        scale_x = W / frame_for_infer.shape[1]
        scale_y = H / frame_for_infer.shape[0]

        if r.masks is not None and len(r.masks.data) > 0:
            clses = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy()

            for mask, cls_id in zip(masks, clses):
                if cls_id != 0:
                    continue
                m = cv2.resize(mask, (frame_for_infer.shape[1], frame_for_infer.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8)
                if MASK_DILATE_KERNEL is not None:
                    m = cv2.dilate(m, MASK_DILATE_KERNEL, iterations=1)
                m_full = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                if not any_inside_zone and mask_overlaps_zone(m_full, ZONE_POLY):
                    any_inside_zone = True
                if not line_crossed and line_overlaps_mask(m_full, P1, P2, TRIP_LINE_THICK_PX):
                    line_crossed = True

        if r.boxes is not None and len(r.boxes) > 0:
            for bb, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if cls_id != 0:
                    continue
                x1,y1,x2,y2 = bb
                x1,y1,x2,y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 1)

    if USE_HANDS and run_yolo_now:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            line_mask = make_line_mask((H, W), P1, P2, TRIP_LINE_THICK_PX)
            for hlm in res.multi_hand_landmarks:
                pts = [(int(lm.x * W), int(lm.y * H)) for lm in hlm.landmark]
                if not any_inside_zone and any(point_in_polygon(p, ZONE_POLY) for p in pts):
                    any_inside_zone = True
                if not line_crossed and any(line_mask[min(max(0, py), H-1), min(max(0, px), W-1)] for (px,py) in pts):
                    line_crossed = True
                if any_inside_zone and line_crossed:
                    break

    triggered = any_inside_zone or line_crossed
    if triggered:
        play_alert()
        cv2.putText(frame, "ALERT: HUMAN INTRUSION!", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if line_crossed:
            maybe_pulse_power_cut()
    else:
        stop_alert()

    if relay_on and time.time() >= relay_off_at:
        power_relay_off()

    # ======== POWER STATUS OVERLAY ========
    if relay_on:
        power_text = "POWER OFF (RELAY ACTIVE)"
        power_color = (0, 0, 255)
    else:
        power_text = "POWER ON"
        power_color = (0, 255, 0)
    cv2.putText(frame, power_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, power_color, 2)

    # FPS overlay
    t = time.time()
    fps = 1.0 / max(1e-6, (t - last_t))
    last_t = t
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow(win_name, frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

# Cleanup
stop_alert()
if pygame_inited:
    try: pygame.mixer.quit()
    except Exception: pass
if USE_HANDS and hands is not None:
    hands.close()
if gpio_ok:
    try:
        power_relay_off()
        GPIO.cleanup()
    except Exception:
        pass
cap.release()
cv2.destroyAllWindows()
