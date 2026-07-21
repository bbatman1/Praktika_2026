import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# Целевые классы COCO: person(0), car(2), truck(5), bus(7)
TARGET_IDS = [0, 2, 5, 7]
COLORS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "truck": (0, 165, 255),
    "bus": (0, 255, 255),
}

def detect_image(model, image_path, conf=0.25, save=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot load {image_path}")
        return
    results = model(img, conf=conf, classes=TARGET_IDS, verbose=False)
    print(f"Detected: {len(results[0].boxes)} objects")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf_val:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"  {cls_name}: conf={conf_val:.2f}, bbox=({x1},{y1},{x2},{y2})")
    if save:
        out = "result_" + Path(image_path).name
        cv2.imwrite(out, img)
        print(f"Saved: {out}")
    return img

def detect_video(model, source=0, conf=0.25):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return
    fps_list = []
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=conf, classes=TARGET_IDS, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf_val:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        end = time.time()
        fps = 1.0 / (end - start) if end > start else 0
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()
    if fps_list:
        print(f"Avg FPS: {sum(fps_list)/len(fps_list):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)
    src = str(args.source)
    if src.isdigit():
        detect_video(model, source=int(src), conf=args.conf)
    else:
        detect_image(model, src, conf=args.conf)