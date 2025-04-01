from ultralytics import YOLO
import os

# Step 3: Define paths for the dataset
dataset_path = r"dataset2"  # Change this to the path where your dataset is stored
train_data = os.path.join(dataset_path, 'train')  # Training data directory
val_data = os.path.join(dataset_path, 'val')  # Validation data directory
yaml_path = os.path.join(dataset_path, 'moon.yaml')  # YAML file describing dataset



# Step 5: Configure and train the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use other model sizes like yolov8s.pt, yolov8m.pt, etc.

model.train(
    data=yaml_path,
    epochs=100,  # Number of epochs
    imgsz=900,  # Image size
    batch=16,  # Batch size
    name='custom_yolov8_model'  # Name for the training run
)

# Step 6: Evaluate the model
metrics = model.val(data=yaml_path)

# Step 7: Save the trained model
#model_path = 'best.pt'  # Path where the model will be saved
#model.save(model_path)

#print(f"Model saved to {model_path}")
print(f"Metrics: {metrics}")
