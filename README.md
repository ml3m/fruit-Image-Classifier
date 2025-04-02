# Fruit Image Classification with MobileNetV2

A deep learning model for accurate classification of fruit images using MobileNetV2 architecture with TensorFlow.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="logo.png" alt="Logo" width="400" height="400">

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a fruit image classification system using MobileNetV2, a lightweight deep learning architecture suitable for mobile and edge devices. The model is trained on the Fruit Images Dataset and can classify fruits across multiple categories with high accuracy. The implementation features a two-phase training approach with initial training followed by fine-tuning for optimal performance.

## Dataset

### Fruit Images Dataset

This model is trained on the Fruit Images Dataset, which contains high-quality images of various fruits organized by categories.

**Dataset Details:**
- 120+ fruit categories
- ~42,000 training images
- ~17,000 test images
- Images sized at various dimensions (resized to 224x224 during processing)

**Download Instructions:**
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
2. Extract the contents to `data/fruits/` in the project directory
3. Ensure the following directory structure:
   ```
   data/fruits/
   ├── Fruit-Images-Dataset-master/
       ├── Training/
       │   ├── Apple Golden 1/
       │   ├── Apple Golden 2/
       │   ├── Apple Golden 3/
       │   └── ...
       ├── Test/
           ├── Apple Golden 1/
           ├── Apple Golden 2/
           ├── Apple Golden 3/
           └── ...
   ```

**Citation:**
Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Matplotlib
- GPU with CUDA support (recommended for faster training)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fruit-image-classification.git
   cd fruit-image-classification
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv fruit-env
   source fruit-env/bin/activate  # On Windows: fruit-env\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Model Architecture

The model architecture consists of:

1. **Base Model**: MobileNetV2 pre-trained on ImageNet
2. **Custom Classification Head**:
   - Global Average Pooling
   - Dense layer (256 units, ReLU activation)
   - Dropout (0.3) for regularization
   - Output layer with softmax activation (number of classes = number of fruit categories)

**Training Strategy**:
- Phase 1: Train only the custom classification head (5 epochs)
- Phase 2: Fine-tune the model by unfreezing the top 20 layers of MobileNetV2 (5 epochs)

## Usage

### Training

To train the model from scratch:

```bash
python fine_tune.py
```

This will:
- Load and preprocess the Fruit Images Dataset
- Create and compile the MobileNetV2 model
- Train the model in two phases (initial training and fine-tuning)
- Save checkpoints after each epoch
- Save the best model based on validation accuracy
- Generate training history plots
- Save the final model

Training parameters can be modified in the script:
- `img_width, img_height`: Image dimensions (default: 224x224)
- `batch_size`: Batch size (default: 64)
- `epochs`: Total epochs (default: 10, split into 5+5)

### Prediction

To classify a single fruit image:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the trained model
model = load_model('best_model.h5')

# Load class indices
class_indices = {}
with open('class_indices.txt', 'r') as f:
    for line in f:
        fruit, idx = line.strip().split(': ')
        class_indices[int(idx)] = fruit

# Preprocess the image
img_path = 'path/to/your/fruit/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"Predicted fruit: {class_indices[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
```

## Results

The model achieves the following performance metrics on the test dataset:

- **Accuracy**: ~95% on the test set
- **Training Time**: Approximately 2-3 hours on a standard GPU
- **Model Size**: ~14MB (suitable for mobile deployment)

![Training History](https://via.placeholder.com/600x300.png?text=Training+and+Validation+Accuracy/Loss)

## Project Structure

```
fruit-image-classification/
├── data/
│   └── fruits/                   # Dataset directory
├── checkpoints/                  # Saved model checkpoints
├── logs/                         # TensorBoard logs
├── fine_tune.py                  # Training script
├── evaluate.py                   # Evaluation script
├── predict.py                    # Prediction script
├── best_model.h5                 # Best model (highest validation accuracy)
├── fruit_model.h5                # Final trained model
├── class_indices.txt             # Class mapping file
├── training_history.png          # Training metrics visualization
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Future Improvements

- [ ] Implement data augmentation techniques to improve model robustness
- [ ] Explore other lightweight architectures (EfficientNet, MobileNetV3)
- [ ] Add support for TensorFlow Lite conversion for mobile deployment
- [ ] Create a simple web interface for online fruit classification
- [ ] Implement explainable AI techniques to visualize model decision-making
- [ ] Add support for fruit detection (not just classification)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [Fruit Images Dataset](https://www.kaggle.com/datasets/moltean/fruits) creators for providing the dataset
- TensorFlow team for the MobileNetV2 implementation
- The research community for advancements in efficient deep learning models

---

**Note**: This project is for educational purposes. For commercial
applications, please ensure you comply with the dataset's license terms.
