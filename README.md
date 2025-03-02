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
Users can provide feedback via “Correct”/“Incorrect” buttons for uncertain photos (confidence < 0.7), logged in `feedback.csv` (e.g., `photo_1,True`, `photo_4,True`).


## Validation Results (Example)
Button to validate Correct/Incorrect to images
- **Left**: Suitable: Yes (Confidence: 0.82) – Clear, correct left-side view.
- **Right**: Suitable: No (Confidence: 0.68) – Flagged for human review (feedback needed).
- **Front**: Suitable: Yes (Confidence: 0.90) – Clear, frontal view.
- **Back**: Suitable: Yes (Confidence: 0.78) – Clear, rear view.
- **Fuel Gauge**: Suitable: No (Confidence: 0.58) – Flagged for human review (feedback needed).

## Tech Stack
- **Back-End**: Python, Flask, PyTorch, OpenCV
- **Front-End**: JavaScript, HTML, CSS

Damage Highlight:
![image](https://github.com/user-attachments/assets/610b1521-8d59-445e-9ce2-a91fa0307b30)

