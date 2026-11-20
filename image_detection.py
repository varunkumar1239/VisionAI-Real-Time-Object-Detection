from ultralytics import YOLO

print("STEP 1: Loading YOLO model...")

model = YOLO("yolo26n.pt")

print("STEP 2: Running object detection...")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    show=True,
    verbose=True
)

print("\nSTEP 3: Detection results")

for result in results:
    print("Image:", result.path)

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = result.names[class_id]

            print(
                f"Detected: {object_name} "
                f"| Confidence: {confidence:.2%}"
            )
    else:
        print("No objects detected.")

    print("Output folder:", result.save_dir)

print("\nSUCCESS: Detection process finished!")