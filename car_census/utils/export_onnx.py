from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("weights/yolo26s.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo26s.onnx'

# Load the exported ONNX model
onnx_model = YOLO("weights/yolo26s.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")