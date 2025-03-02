# Rental Photo Validator
A hackathon project to validate rental machinery return photos and detect damage.

## Features
- Validates 5 photos (Left, Right, Front, Back, Fuel Gauge) for suitability.
- Detects damage by comparing before/after photos.
- Logs human feedback to improve accuracy.

## Tech Stack
- **Back-End**: Python, Flask, PyTorch, OpenCV
- **Front-End**: JavaScript, HTML, CSS

## Setup
1. Install dependencies: `pip install flask torch torchvision opencv-python pillow numpy`
2. Run: `python app.py`
3. Visit: `http://127.0.0.1:5000`

## Demo
- **Validation**: Upload 5 JCB photos (e.g., excavator, loader) to see suitability results.
- **Damage**: Compare before/after pairs to highlight damage.
- **Feedback**: Correct uncertain predictions to improve the model.

![image](https://github.com/user-attachments/assets/b12b9314-6c3b-4d54-b808-5f64d91210cc)
