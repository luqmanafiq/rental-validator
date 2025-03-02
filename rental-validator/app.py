import os
import tempfile
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from rental_photo_validator import RentalPhotoValidator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'rental_validator_model.h5'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize validator
validator = RentalPhotoValidator(model_path=app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/damage_detection/<filename>')
def damage_detection_file(filename):
    return send_from_directory('damage_detection', filename)

@app.route('/api/validate', methods=['POST'])
def validate_photos():
    if 'front' not in request.files or 'back' not in request.files or \
       'left_side' not in request.files or 'right_side' not in request.files or \
       'fuel_gauge' not in request.files:
        return jsonify({'error': 'Missing required photos'}), 400
    
    photos = {}
    for category in ['front', 'back', 'left_side', 'right_side', 'fuel_gauge']:
        file = request.files[category]
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{category}_{uuid.uuid4().hex}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            photos[category] = filepath
    
    results = validator.validate_return_photos(photos)
    return jsonify(results)

@app.route('/api/compare', methods=['POST'])
def compare_photos():
    if 'before' not in request.files or 'after' not in request.files or 'category' not in request.form:
        return jsonify({'error': 'Missing required files or category'}), 400
    
    before_file = request.files['before']
    after_file = request.files['after']
    category = request.form['category']
    
    if before_file and after_file and allowed_file(before_file.filename) and allowed_file(after_file.filename):
        before_filename = secure_filename(f"before_{uuid.uuid4().hex}.jpg")
        after_filename = secure_filename(f"after_{uuid.uuid4().hex}.jpg")
        
        before_filepath = os.path.join(app.config['UPLOAD_FOLDER'], before_filename)
        after_filepath = os.path.join(app.config['UPLOAD_FOLDER'], after_filename)
        
        before_file.save(before_filepath)
        after_file.save(after_filepath)
        
        results = validator.compare_before_after(before_filepath, after_filepath, category)
        return jsonify(results)
    
    return jsonify({'error': 'Invalid files'}), 400

@app.route('/api/uncertain', methods=['GET'])
def get_uncertain_samples():
    samples = validator.get_uncertain_samples()
    return jsonify(samples)

@app.route('/api/uncertain/resolve', methods=['POST'])
def resolve_uncertain():
    if 'image_path' not in request.form or 'correct_category' not in request.form:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    image_path = request.form['image_path']
    correct_category = request.form['correct_category']
    
    result = validator.resolve_uncertain_sample(image_path, correct_category)
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def train_model():
    data_dir = request.form.get('data_dir', 'learning_samples')
    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 32))
    
    if not os.path.exists(data_dir):
        return jsonify({'error': f'Data directory {data_dir} not found'}), 400
    
    try:
        history = validator.train(data_dir, epochs, batch_size)
        validator.save_model(app.config['MODEL_PATH'])
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained for {epochs} epochs and saved',
            'accuracy': float(history.history['accuracy'][-1]),
            'validation_accuracy': float(history.history['val_accuracy'][-1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threshold', methods=['POST'])
def update_threshold():
    if 'threshold' not in request.form:
        return jsonify({'error': 'Missing threshold parameter'}), 400
    
    try:
        threshold = float(request.form['threshold'])
        if validator.update_confidence_threshold(threshold):
            return jsonify({'status': 'success', 'new_threshold': threshold})
        else:
            return jsonify({'error': 'Invalid threshold value. Must be between 0 and 1'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid threshold format'}), 400

if __name__ == '__main__':
    app.run(debug=True)