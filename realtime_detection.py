
import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Webcam not found")
    exit()

# Get webcam dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video recording settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "detection_output.mp4",
    fourcc,
    20.0,
    (width, height)
)

prev_time = time.time()

print("Real-Time Detection Started")
print("Press Q to stop recording")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO object tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=416,
        conf=0.35,
        verbose=False
    )

    # Draw bounding boxes and IDs
    output = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 0.001)
    prev_time = current_time

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Save processed frame
    out.write(output)

    # Display output
    cv2.imshow("YOLO Detection and Recording", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as detection_output.mp4")
