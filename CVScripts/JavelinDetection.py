from ultralytics import YOLO
import cv2

def detect_javelin_on_image(model_path: str, image_path: str, target_class_name: str = "javelin", conf_thresh: float = 0.25):
    model = YOLO(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model.predict(source=img, conf=conf_thresh, verbose=False)
    r = results[0]

    class_id_to_name = r.names
    out = []

    if r.boxes is None or len(r.boxes) == 0:
        return out

    for b in r.boxes:
        cls_id = int(b.cls.item())
        cls_name = class_id_to_name.get(cls_id, str(cls_id))

        if cls_name != target_class_name:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf.item())

        out.append({
            "class": cls_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return out

if __name__ == "__main__":
    boxes = detect_javelin_on_image(
        model_path="runs/detect/train/weights/best.pt",  # change to your model
        image_path="test.jpg",
        target_class_name="javelin",
        conf_thresh=0.25
    )
    print(boxes)
