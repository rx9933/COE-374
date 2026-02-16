import time
from pathlib import Path
import cv2
from ultralytics import YOLO

#use pip install opencv-python numpy ultralytics
#change location of images in script
#then run python yolo_test.py

images_dir = Path(r"C:\Users\gauta\OneDrive\Documents\MATLAB\Pretrained-YOLOv8-Network-For-Object-Detection-main\images")

img_paths = [
    images_dir / "IMG_8015.jpeg",
    images_dir / "IMG_8016.jpeg",
    images_dir / "IMG_8017.jpeg",
]

out_dir = Path("tracked_out")
out_dir.mkdir(exist_ok=True)

#yolo settings
model = YOLO("yolov8s.pt")

CONF = 0.20      # raise this to reduce background 
IMGSZ = 1080 # inference resolution

# Only keep these classes on screen (no cars)
KEEP = {"person", "backpack", "bottle", "sports ball"}

# Rename mapping
RENAME = {"sports ball": "shotput"}




_ = model.predict(str(img_paths[0]), conf=CONF, imgsz=IMGSZ, verbose=False)

times = []

for i, p in enumerate(img_paths, start=1):
    if not p.exists():
        raise FileNotFoundError(f"Missing image: {p}")

    # time_inference
    t0 = time.perf_counter()
    results = model.predict(str(p), conf=CONF, imgsz=IMGSZ, verbose=False)
    t1 = time.perf_counter()

    dt = t1 - t0
    times.append(dt)

    r = results[0]

    # original image
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"cv2 could not read: {p}")

    # draw ONLY filtered detections
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            name = r.names[cls_id]
            conf = float(b.conf.item())

            if name not in KEEP:
                continue

            label = RENAME.get(name, name)

            # xyxy coords
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

            # Draw box
            if label == "shotput":
                color = (0, 0, 255)  
            else:
                color = (255, 0, 0) 

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # Write timing on the frame
    cv2.putText(
        img,
        f"inference: {dt*1000:.1f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4
    )
    cv2.putText(
        img,
        f"inference: {dt*1000:.1f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    out_path = out_dir / f"frame_{i:02d}_yolo_filtered.png"
    cv2.imwrite(str(out_path), img)

    print(f"Saved: {out_path} | time={dt:.4f}s ({1/dt:.2f} FPS equiv)")

# Summary timing
avg = sum(times) / len(times)
print("\n--- Timing Summary ---")
print(f"Avg per image: {avg*1000:.1f} ms")
print(f"Avg FPS equiv: {1/avg:.2f}")
print(f"Min/Max: {min(times)*1000:.1f} / {max(times)*1000:.1f} ms")
print(f"Outputs in: {out_dir.resolve()}")

