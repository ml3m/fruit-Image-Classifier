import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

def predict_with_model():
    # Check which model file exists
    if os.path.exists('best_model.h5'):
        model_path = 'best_model.h5'
        print("Using best_model.h5")
    elif os.path.exists('fruit_model.h5'):
        model_path = 'fruit_model.h5'
        print("Using fruit_model.h5")
    else:
        print("ERROR: No model file found! Looking for best_model.h5 or fruit_model.h5")
        return

    # Load the model
    model = load_model(model_path)
    
    # Load class indices
    class_indices = {}
    try:
        with open('class_indices.txt', 'r') as f:
            for line in f:
                fruit, idx = line.strip().split(': ')
                class_indices[int(idx)] = fruit
        print(f"Loaded {len(class_indices)} fruit classes")
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return
    
    # Ask for image path
    img_path = input("Enter the path to your fruit image: ")
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found.")
        return
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_indices[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    
    # Show top 5 predictions
    print("\nTop 5 Predictions:")
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    for i, idx in enumerate(top_indices):
        fruit = class_indices[idx]
        conf = predictions[0][idx] * 100
        print(f"{i+1}. {fruit}: {conf:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    predict_with_model()
