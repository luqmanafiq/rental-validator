# Rental Photo Validator

## Hackathon Project Overview
A 24-hour hackathon solution to automate the validation of rental machinery return photos and detect damage using machine learning and computer vision.

## The Challenge
Rental companies struggle with efficiently processing returned equipment photos:
- Manual validation is time-consuming
- Damage assessment is subjective
- Processing delays affect customer satisfaction
- Photo quality issues complicate inspections

## Our Solution
We built a web application that uses computer vision and machine learning to:
1. Validate photo quality and angles
2. Detect potential damage by comparing before/after images
3. Learn from human feedback

## Key Features
- **5-Point Inspection**: Validates critical equipment views (Left, Right, Front, Back, Fuel Gauge)
- **Damage Detection**: Highlights differences between before/after photos
- **Feedback System**: Improves accuracy through human validation
- **Confidence Scoring**: Provides reliability metrics for each prediction

## Tech Stack
- **Back-End**: Python, Flask, PyTorch, OpenCV
- **Front-End**: JavaScript, HTML, CSS
- **Image Processing**: NumPy, PIL

## Demo

### Quick Start
1. Install dependencies: `pip install flask torch torchvision opencv-python pillow numpy`
2. Run the app: `python app.py`
3. Open: `http://127.0.0.1:5000`

### Validation Demo
Upload 5 JCB machinery photos to see results like:
- **Left**: Suitable: Yes (Confidence: 0.82)
- **Right**: Suitable: No (Confidence: 0.68) - Flagged for review
- **Front**: Suitable: Yes (Confidence: 0.90)
- **Back**: Suitable: Yes (Confidence: 0.78)
- **Fuel Gauge**: Suitable: No (Confidence: 0.58) - Flagged for review

### Damage Detection
Compare before/after photos to highlight damage:

![Damage Highlight](https://github.com/user-attachments/assets/610b1521-8d59-445e-9ce2-a91fa0307b30)

### Human Feedback
For uncertain predictions (confidence < 0.7):
- Click "Correct" or "Incorrect" to improve the model
- Feedback is logged in `feedback.csv`

## Implementation Details

### Machine Learning Implementation
- Used pre-trained ResNet18 model adapted for photo validation
- Simple confidence threshold for review flagging
- Image difference algorithm for damage detection

### Rapid Development Approaches
- Leveraged pre-trained models for quick implementation
- Simple file-based feedback system
- Responsive design with minimal dependencies

## Hackathon Results
- **Completed in**: 6 days
- **Team members**: Me
- **Awards/Recognition**: Participation (Only winner are announced)

## Future Potential
If developed further, the system could:
- Train on rental-specific image dataset
- Implement more sophisticated damage classification
- Integrate with rental management systems
- Add mobile app for field inspections
- Generate detailed damage reports

## How to Contribute
This hackathon project is open for improvements! To contribute:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## License
This project is open source under the MIT License.

