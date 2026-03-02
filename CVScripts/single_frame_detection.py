"""
YOLO-based shotput detection. Provides a callable that takes an image and returns
the shotput bounding box.
"""
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

model = None

DEFAULT_CONF = 0.20      # raise this to reduce background 
DEFAULT_IMGSZ = 1080 # inference resolution
SHOTPUT_CLASS_NAMES = {"sports ball"}  # Rename mapping


def getmodel():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("yolov8s.pt")
    return model


def detect_shotput(image, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, model=None):
    """
    Run YOLO on an image and return the shotput bounding box (if detected).

    Args:
        image: BGR image (H, W, 3), e.g. from cv2.imread() or a video frame.
        conf: Confidence threshold (0–1). Lower = more detections, more false positives.
        imgsz: Inference resolution (YOLO resizes internally).
        model: Optional pre-loaded YOLO model. If None, uses default yolov8s.pt.

    Returns:
        If a shotput (sports ball) is detected return ((x1, y1, x2, y2), confidence) else None.
    """
    if image is None or not isinstance(image, np.ndarray):
        print("Image is None or not an numpy array")
        return None
    if model is None:
        print("Model is None, loading default model")
        model = getmodel()

    results = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("No shotput detected")
        return None

    #finding the best shotput match and location
    r = results[0]
    best = None
    best_conf = 0.0
    for b in r.boxes:
        cls_id = int(b.cls.item())
        name = r.names[cls_id]
        if name not in SHOTPUT_CLASS_NAMES:
            continue
        c = float(b.conf.item())
        if c > best_conf:
            best_conf = c
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            best = ((x1, y1, x2, y2), c)

    return best


for i, p in enumerate(img_paths, start=1):
    if not p.exists():
        raise FileNotFoundError(f"Missing image: {p}")


# ---------------------------------------------------------------------------
# -------------------------------- Demo -------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    images_dir = Path(r"C:\Users\gauta\OneDrive\Documents\MATLAB\Pretrained-YOLOv8-Network-For-Object-Detection-main\images")
    img_paths = [
        images_dir / "IMG_8015.jpeg",
        images_dir / "IMG_8016.jpeg",
        images_dir / "IMG_8017.jpeg",
    ]
    out_dir = Path("tracked_out")
    out_dir.mkdir(exist_ok=True)

    CONF = 0.20
    IMGSZ = 1080
    times = []

    for i, p in enumerate(img_paths, start=1):
        if not p.exists():
            print(f"Skip (missing): {p}")
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skip (unreadable): {p}")
            continue

        t0 = time.perf_counter()
        result = detect_shotput(img, conf=CONF, imgsz=IMGSZ)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if result:
            (x1, y1, x2, y2), c = result
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"shotput {c:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        dt = times[-1]
        cv2.putText(img, f"inference: {dt*1000:.1f} ms", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out_path = out_dir / f"frame_{i:02d}_yolo_filtered.png"
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path} | time={dt:.4f}s | shotput={'yes' if result else 'no'}")

    if times:
        avg = sum(times) / len(times)
        print(f"\nAvg per image: {avg*1000:.1f} ms ({1/avg:.2f} FPS equiv)")
        print(f"Outputs in: {out_dir.resolve()}")
    else:
        print("No images found")
