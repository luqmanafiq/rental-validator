import os
import cv2
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
import uuid
from datetime import datetime

class RentalPhotoValidator:
    def __init__(self, model_path=None):
        self.categories = ['front', 'back', 'left_side', 'right_side', 'fuel_gauge', 'invalid']
        self.confidence_threshold = 0.7  # Threshold for confident predictions
        self.learning_folder = "learning_samples"
        self.history_file = "validation_history.json"
        self.history = self._load_history()
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.model = torch.load(model_path)
        else:
            print("Creating new model")
            self._create_model()
            
        # Create learning folder if it doesn't exist
        if not os.path.exists(self.learning_folder):
            os.makedirs(self.learning_folder)
            for category in self.categories:
                os.makedirs(os.path.join(self.learning_folder, category), exist_ok=True)
    
    def _create_model(self):
        """Create a transfer learning model based on MobileNetV2"""
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.categories))
        
        # Freeze base model layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)
    
    def _load_history(self):
        """Load validation history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"validations": [], "uncertain_samples": [], "damage_detections": []}
    
    def _save_history(self):
        """Save validation history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _preprocess_image(self, image_path):
        """Preprocess image for model input"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension
        return img
    
    def predict_category(self, image_path):
        """Predict the category of a single image"""
        img = self._preprocess_image(image_path)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
        
        predicted_class = self.categories[predicted.item()]
        return predicted_class, confidence
    
    def save_for_learning(self, image_path, category):
        """Save an image for future learning"""
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
            
        # Generate unique filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(self.learning_folder, category, filename)
        
        # Copy image
        img = cv2.imread(image_path)
        cv2.imwrite(save_path, img)
        
        return save_path
    
    def validate_return_photos(self, photos):
        """
        Validate a set of rental return photos
        
        Args:
            photos: Dictionary with keys: 'front', 'back', 'left_side', 'right_side', 'fuel_gauge'
                   and values as image file paths
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "status": "incomplete",
            "missing_categories": [],
            "uncertain_categories": [],
            "valid_categories": [],
            "details": {}
        }
        
        required_categories = ['front', 'back', 'left_side', 'right_side', 'fuel_gauge']
        
        # Check if all required photos are provided
        for category in required_categories:
            if category not in photos or not photos[category]:
                results["missing_categories"].append(category)
        
        # Validate each provided photo
        for category, image_path in photos.items():
            if not image_path or not os.path.exists(image_path):
                continue
                
            predicted_category, confidence = self.predict_category(image_path)
            
            results["details"][category] = {
                "provided_image": image_path,
                "predicted_as": predicted_category,
                "confidence": float(confidence),
                "status": "unknown"
            }
            
            # Determine validation status based on confidence
            if confidence >= self.confidence_threshold:
                if predicted_category == category:
                    results["details"][category]["status"] = "valid"
                    results["valid_categories"].append(category)
                else:
                    results["details"][category]["status"] = "invalid"
                    # Save incorrect prediction for learning
                    self.save_for_learning(image_path, category)
            else:
                results["details"][category]["status"] = "uncertain"
                results["uncertain_categories"].append(category)
                # Save uncertain prediction for human review
                save_path = self.save_for_learning(image_path, "uncertain")
                self.history["uncertain_samples"].append({
                    "image": save_path,
                    "predicted_as": predicted_category,
                    "provided_as": category,
                    "confidence": float(confidence),
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Determine overall status
        if len(results["missing_categories"]) > 0:
            results["status"] = "incomplete"
        elif len(results["uncertain_categories"]) > 0:
            results["status"] = "needs_review"
        elif len(results["valid_categories"]) == len(required_categories):
            results["status"] = "valid"
        else:
            results["status"] = "invalid"
        
        # Update history
        self.history["validations"].append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": results["status"],
            "details": results["details"]
        })
        self._save_history()
        
        return results
    
    def compare_before_after(self, before_image, after_image, category):
        """
        Compare before and after images to detect damage
        Using Structural Similarity Index (SSIM) for simple comparison
        
        Args:
            before_image: Path to the image taken at rental start
            after_image: Path to the image taken at rental return
            category: Category of the images (front, back, etc.)
            
        Returns:
            Dictionary with damage detection results
        """
        if not os.path.exists(before_image) or not os.path.exists(after_image):
            return {"status": "error", "message": "One or both images not found"}
        
        # Load images
        img1 = cv2.imread(before_image)
        img2 = cv2.imread(after_image)
        
        # Resize to same dimensions
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        (score, diff) = cv2.compareSSIM(gray1, gray2, full=True)
        
        # Threshold the difference image
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Find contours in the thresholded difference image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization
        img_copy = img2.copy()
        cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
        
        # Generate unique filename for visualization
        result_filename = f"damage_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join("damage_detection", result_filename)
        
        # Create directory if it doesn't exist
        os.makedirs("damage_detection", exist_ok=True)
        
        # Save visualization
        cv2.imwrite(result_path, img_copy)
        
        # Calculate percentage of different pixels
        damage_percentage = 100 * (1 - score)
        
        # Determine if damage is detected based on threshold
        damage_detected = damage_percentage > 10  # Adjust threshold as needed
        
        # Save to history
        self.history["damage_detections"].append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "category": category,
            "before_image": before_image,
            "after_image": after_image,
            "similarity_score": score,
            "damage_percentage": damage_percentage,
            "damage_detected": damage_detected,
            "visualization": result_path
        })
        self._save_history()
        
        return {
            "status": "success",
            "damage_detected": damage_detected,
            "damage_percentage": damage_percentage,
            "similarity_score": score,
            "visualization": result_path,
            "category": category
        }
    
    def train(self, data_dir, epochs=10, batch_size=32):
        """
        Train the model on new data
        
        Args:
            data_dir: Directory containing categorized images
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Data augmentation
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Training data generator
        train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train model
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
        return {"loss": running_loss/len(train_loader)}
    
    def evaluate(self, test_data_dir):
        """
        Evaluate model performance
        
        Args:
            test_data_dir: Directory containing test images
            
        Returns:
            Evaluation metrics
        """
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate model
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.categories, 
                   yticklabels=self.categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_path": cm_path
        }
    
    def save_model(self, filepath):
        """Save the model to disk"""
        torch.save(self.model, filepath)
        
    def update_confidence_threshold(self, new_threshold):
        """Update the confidence threshold for predictions"""
        if 0 < new_threshold < 1:
            self.confidence_threshold = new_threshold
            return True
        return False
            
    def get_uncertain_samples(self):
        """Get samples that need human review"""
        return self.history["uncertain_samples"]
    
    def resolve_uncertain_sample(self, image_path, correct_category):
        """Resolve an uncertain sample by providing correct category"""
        # Find the uncertain sample in history
        for i, sample in enumerate(self.history["uncertain_samples"]):
            if sample["image"] == image_path:
                # Move image to correct category folder
                if os.path.exists(image_path):
                    new_path = self.save_for_learning(image_path, correct_category)
                    
                    # Update the history entry
                    self.history["uncertain_samples"][i]["resolved"] = True
                    self.history["uncertain_samples"][i]["corrected_category"] = correct_category
                    self.history["uncertain_samples"][i]["resolution_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self._save_history()
                    
                    return {"status": "success", "new_path": new_path}
                else:
                    return {"status": "error", "message": "Image not found"}
        
        return {"status": "error", "message": "Uncertain sample not found in history"}