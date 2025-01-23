from flask import Flask, request, render_template, send_from_directory
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTION_FOLDER'] = 'detections'

model = YOLO('runs/detect/custom_yolov8_model4/weights/best.pt')  # Update the path if necessary

# Ensure the upload and detection directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

def detect_and_plot(image_path, output_path):
    results = model.predict(source=image_path)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    class_names = {0: 'crater', 1: 'boulder'}
    colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Red for boulder, Green for crater
    
    result = results[0]
    
    for box, label in zip(result.boxes.xyxy, result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        color = colors[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    labels = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    
    for box, score, label in zip(result.boxes.xyxy, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{class_names[label]}: {score:.2f}"
        color = colors[label]
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        print("POST request received")
        if 'image' not in request.files:
            print("No file part")
            return 'No file part'
        
        file = request.files['image']
        
        if file.filename == '':
            print("No selected file")
            return 'No selected file'
        
        if file:
            print(f"File received: {file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")
            
            output_path = os.path.join(app.config['DETECTION_FOLDER'], file.filename)
            detect_and_plot(file_path, output_path)
            print(f"Detection completed. Output saved to: {output_path}")
            
            return render_template('result.html', input_image=file.filename, output_image=file.filename)
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detections/<filename>')
def detected_file(filename):
    return send_from_directory(app.config['DETECTION_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
