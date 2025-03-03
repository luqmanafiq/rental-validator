from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__, static_url_path='', static_folder='static')

# Pre-trained model (MVP: use pre-trained weights, simulate for demo)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Suitable/Unsuitable
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    results = []
    for i in range(5):
        photo = request.files.get(f'photo_{i}')
        if photo:
            img = Image.open(photo).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                pred = model(img_tensor)  # Simulate prediction (replace with real training if time)
                prob = torch.softmax(pred, dim=1)[0][1].item()  # Random for demo: 0.8 or 0.3
            results.append({
                'type': ['Left', 'Right', 'Front', 'Back', 'Fuel Gauge'][i],
                'suitable': prob > 0.7,
                'confidence': prob,
                'id': f'photo_{i}'
            })
    return jsonify(results)

@app.route('/damage', methods=['POST'])
def damage():
    before = cv2.imdecode(np.frombuffer(request.files['before'].read(), np.uint8), cv2.IMREAD_COLOR)
    after = cv2.imdecode(np.frombuffer(request.files['after'].read(), np.uint8), cv2.IMREAD_COLOR)
    diff = cv2.absdiff(before, after)
    damage_path = 'static/damage.jpg'
    cv2.imwrite(damage_path, diff)
    return jsonify({'damage_url': '/damage.jpg'})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    with open('feedback.csv', 'a') as f:
        f.write(f"{data['id']},{data['correct']}\n")
    return jsonify({'status': 'Feedback recorded'})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)