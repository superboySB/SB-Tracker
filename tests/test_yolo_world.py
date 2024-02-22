from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt

# Define custom classes
model.set_classes(["person", "bus"])

# Execute prediction for specified categories on an image
results = model.predict('data/bus.jpg')

# Optionally, save the image with detections
results[0].save("data/detected_bus.jpg")
