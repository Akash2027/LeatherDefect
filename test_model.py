from ultralytics import YOLO

model = YOLO("modelv1.pt")

results = model.predict(
    source="testdataset/hole1_21.jpg",
    save=True,
    conf=0.25
)

print("Prediction completed!")