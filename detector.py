


from ultralytics import YOLO
import cv2

# Load the YOLO model (using a lightweight model)
model = YOLO('yolov8n.pt')

def detect_objects(frame):
    results = model(frame, show=False)  # Run inference
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Draw rectangle and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[cls]}: {conf:.2f}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame