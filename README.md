# YOLOv8 Object Detection

This project demonstrates real-time object detection using YOLOv8 and OpenCV.

## Installation

First, install the required dependencies:

```sh
pip install opencv-python numpy torch torchvision ultralytics
```

## Running the Object Detection Script

1. Ensure you have a webcam connected.
2. Run the script:

```sh
python object_detection.py
```

Press `q` to exit the program.

## object_detection.py

```python
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the nano version for speed

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```
