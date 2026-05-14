"""
Detect if an image is Real or AI-generated
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os

class Detector:
    def __init__(self):
        """Load the trained model"""
        if os.path.exists('models/detector.h5'):
            self.model = tf.keras.models.load_model('models/detector.h5')
            print("✅ Model loaded")
        else:
            self.model = None
            print("❌ Train model first: python train.py")
    
    def detect(self, image_path):
        """Detect image: Returns (result, confidence)"""
        if self.model is None:
            return "No model", 0
        
        # Load and prepare image
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            return "AI-GENERATED", prediction * 100
        else:
            return "REAL", (1 - prediction) * 100

# Simple command line interface
if __name__ == "__main__":
    detector = Detector()
    
    if detector.model:
        print("\nEnter image path to detect (or 'quit' to exit)")
        while True:
            path = input("\nImage path: ").strip().strip('"')
            if path.lower() == 'quit':
                break
            if os.path.exists(path):
                result, conf = detector.detect(path)
                print(f"Result: {result}")
                print(f"Confidence: {conf:.1f}%")
            else:
                print("File not found!")