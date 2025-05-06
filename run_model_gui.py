import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QProgressBar, QFrame, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont, QDrag, QPainter, QColor
from PyQt5.QtCore import Qt, QMimeData, QSize, QThread, pyqtSignal
import io
from PIL import Image, ImageQt

class PredictionThread(QThread):
    prediction_complete = pyqtSignal(list, str)
    
    def __init__(self, model, img_path):
        super().__init__()
        self.model = model
        self.img_path = img_path
        
    def run(self):
        try:
            # Load and preprocess the image
            img = image.load_img(self.img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(processed_img)
            
            # Emit results signal
            self.prediction_complete.emit(predictions[0].tolist(), self.img_path)
        except Exception as e:
            print(f"Error in prediction thread: {e}")


class DropArea(QLabel):
    image_dropped = pyqtSignal(str)  # Add a signal
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n🍎 Drop your fruit image here 🍌\n\nor click to browse files\n\n")
        self.setFont(QFont("Arial", 14))
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f8f8f8;
                padding: 30px;
                color: #666;
            }
            QMainWindow {
                background-color: #f5f7fa;
            }
            QWidget {
                font-family: Arial;
            }
        """)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #3498db;
                    border-radius: 10px;
                    background-color: #e8f6ff;
                    padding: 30px;
                    color: #3498db;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f8f8f8;
                padding: 30px;
                color: #666;
            }
        """)
            
    def dropEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f8f8f8;
                padding: 30px;
                color: #666;
            }
        """)
        
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_dropped.emit(file_path)  # Emit signal instead of calling parent method
                    break
                    
    def mousePressEvent(self, event):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)'
        )
        if file_path:
            self.image_dropped.emit(file_path)  # Emit signal instead of calling parent method


class ResultBar(QWidget):
    def __init__(self, fruit_name, probability, parent=None):
        super().__init__(parent)
        self.fruit_name = fruit_name
        self.probability = probability
        
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 8, 5, 8)
        layout.setSpacing(15)  # Increase spacing between labels and bars
        
        # Create a frame for the name label with background
        name_frame = QFrame()
        name_frame.setFrameShape(QFrame.StyledPanel)
        name_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 3px;
            }
        """)
        name_layout = QHBoxLayout(name_frame)
        name_layout.setContentsMargins(10, 3, 10, 3)
        
        # Fruit name label with improved styling
        self.name_label = QLabel(fruit_name)
        self.name_label.setFont(QFont("Arial", 11, QFont.Bold))  # Make bold
        self.name_label.setStyleSheet("color: #333333;")  # Darker text color
        self.name_label.setAlignment(Qt.AlignCenter)
        name_layout.addWidget(self.name_label)
        
        # Set fixed width for name frame
        name_frame.setFixedWidth(180)
        layout.addWidget(name_frame)
        
        # Progress bar with improved styling
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(int(probability * 100))
        self.progress.setTextVisible(True)
        self.progress.setFormat(f"{probability:.1%}")
        self.progress.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Color based on probability
        color = self.get_color_for_probability(probability)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border-radius: 7px;
                background-color: #f0f0f0;
                text-align: center;
                height: 30px;
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid #dddddd;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 6px;
            }}
        """)
        
        layout.addWidget(self.progress)
        
    def get_color_for_probability(self, prob):
        if prob > 0.8:
            return "#27ae60"  # Green for high probability
        elif prob > 0.5:
            return "#f39c12"  # Orange for medium
        else:
            return "#95a5a6"  # Gray for low
        

############################################################
class FruitClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Classifier")
        self.setMinimumSize(1600, 1000)
        self.setStyleSheet("background-color: white;")
        
        # Initialize properties
        self.model = None
        self.model_path = None
        self.class_indices = {}
        
        # Load model and setup UI
        self.loadModel()
        self.initUI()
    
    def loadModel(self):
        # Load model (try both filenames)
        if os.path.exists('best_model.h5'):
            self.model_path = 'best_model.h5'
        elif os.path.exists('fruit_model.h5'):
            self.model_path = 'fruit_model.h5'
        else:
            print("No model file found")
            self.model = None
            return
        
        print(f"Attempting to load model from {self.model_path}")
        
        # Try to enable mixed precision
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled")
        except Exception as e:
            print(f"Mixed precision not supported: {e}")
        
        # Method 1: Try with custom Cast layer
        try:
            from tensorflow.keras.utils import custom_object_scope
            
            # Define a simple cast function for custom objects
            def cast_function(x, dtype):
                return tf.cast(x, dtype)
            
            with custom_object_scope({'Cast': cast_function}):
                self.model = load_model(self.model_path, compile=False)
                
            # Recompile the model
            self.model.compile(optimizer='adam', 
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
            
            print("Model loaded successfully with method 1")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            
            # Method 2: Try simpler approach
            try:
                print("Trying alternative loading method...")
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={'Cast': lambda x, dtype: tf.cast(x, dtype)}
                )
                print("Model loaded successfully with method 2")
                
            except Exception as e2:
                print(f"All loading methods failed. Error: {e2}")
                self.model = None
                return
        
        # Load class indices
        try:
            with open('class_indices.txt', 'r') as f:
                for line in f:
                    fruit, idx = line.strip().split(': ')
                    self.class_indices[int(idx)] = fruit
            print(f"Loaded {len(self.class_indices)} class labels")
        except Exception as e:
            print(f"Error loading class indices: {e}")
            self.class_indices = {}
    
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("🍓 Fruit Classifier 🥝")
        header.setFont(QFont("Arial", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            color: #2c3e50; 
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 10px;
        """)
        main_layout.addWidget(header)
        
        # Subtitle
        subtitle = QLabel("Drop or select an image to identify the fruit")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        main_layout.addWidget(subtitle)
        
        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left panel - Drop area and image display
        left_panel = QVBoxLayout()
        
        # Drop area for images
        self.drop_area = DropArea(self)
        self.drop_area.image_dropped.connect(self.load_image)
        left_panel.addWidget(self.drop_area)
        
        # Image display (initially hidden)
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 300)
        self.image_display.setMaximumSize(400, 400)
        self.image_display.setStyleSheet("""
            QLabel {
                border: 2px solid #dddddd;
                border-radius: 15px;
                padding: 5px;
                background-color: white;
            }
        """)
        self.image_display.setScaledContents(True)
        self.image_display.hide()
        left_panel.addWidget(self.image_display)
        
        # "Try Another" button (initially hidden)
        self.try_another_btn = QPushButton("Try Another Image")
        self.try_another_btn.setFont(QFont("Arial", 11))
        self.try_another_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.try_another_btn.clicked.connect(self.reset_ui)
        self.try_another_btn.hide()
        left_panel.addWidget(self.try_another_btn)
        
        content_layout.addLayout(left_panel)
        
        # Right panel - Results
        right_panel = QVBoxLayout()
        
        # Results header
        self.results_header = QLabel("Results")
        self.results_header.setFont(QFont("Arial", 18, QFont.Bold))
        self.results_header.setStyleSheet("""
            color: #2c3e50; 
            margin-bottom: 15px;
            padding: 5px 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 5px solid #3498db;
        """)
        self.results_header.hide()
        right_panel.addWidget(self.results_header)
        
        # Main prediction
        self.main_prediction = QLabel()
        self.main_prediction.setFont(QFont("Arial", 14))
        self.main_prediction.setStyleSheet("color: #27ae60; margin-bottom: 20px;")
        self.main_prediction.setWordWrap(True)
        self.main_prediction.hide()
        right_panel.addWidget(self.main_prediction)
        
        # Results container with scroll area
        self.results_container = QFrame()
        self.results_container.setFrameShape(QFrame.NoFrame)
        self.results_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(8)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for results
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setWidget(self.results_container)
        self.results_scroll.setFrameShape(QFrame.NoFrame)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_scroll.setMinimumWidth(350)
        self.results_scroll.hide()
        
        right_panel.addWidget(self.results_scroll)
        right_panel.addStretch()
        
        # Status message for no model
        if self.model is None:
            self.status_label = QLabel("⚠️ No model found. Please place 'best_model.h5' or 'fruit_model.h5' in the same directory.")
            self.status_label.setFont(QFont("Arial", 12))
            self.status_label.setStyleSheet("color: #e74c3c; margin: 20px;")
            self.status_label.setWordWrap(True)
            self.status_label.setAlignment(Qt.AlignCenter)
            right_panel.addWidget(self.status_label)
        
        content_layout.addLayout(right_panel)
        main_layout.addLayout(content_layout)
        
        # Bottom credits
        credits = QLabel("Powered by TensorFlow and ResNet50")
        credits.setFont(QFont("Arial", 9))
        credits.setAlignment(Qt.AlignCenter)
        credits.setStyleSheet("color: #95a5a6; margin-top: 20px;")
        main_layout.addWidget(credits)
        
        self.setCentralWidget(main_widget)
    
    def load_image(self, file_path):
        if self.model is None:
            return
            
        # Hide drop area
        self.drop_area.hide()
        
        # Display the image
        pixmap = QPixmap(file_path)
        self.image_display.setPixmap(pixmap)
        self.image_display.show()
        
        # Show try again button
        self.try_another_btn.show()
        
        # Show loading message
        self.results_header.setText("Processing...")
        self.results_header.show()
        self.main_prediction.setText("Analyzing image...")
        self.main_prediction.show()
        
        # Start prediction in a separate thread
        self.prediction_thread = PredictionThread(self.model, file_path)
        self.prediction_thread.prediction_complete.connect(self.update_results)
        self.prediction_thread.start()
    
    def update_results(self, predictions, img_path):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Convert predictions to list if it's not already
        if not isinstance(predictions, list):
            predictions = predictions.tolist()
            
        # Get top indices
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        # Update main prediction
        top_class_idx = top_indices[0]
        top_class = self.class_indices.get(top_class_idx, f"Class {top_class_idx}")
        confidence = predictions[top_class_idx] * 100
        
        # Show results header
        self.results_header.setText("Results")
        
        # Format the main prediction text
        if confidence > 90:
            confidence_text = "high confidence"
            color = "#27ae60"
        elif confidence > 70:
            confidence_text = "medium confidence"
            color = "#f39c12"
        else:
            confidence_text = "low confidence"
            color = "#e74c3c"
            
        self.main_prediction.setText(f"This appears to be a <b>{top_class}</b> with {confidence_text}.")
        self.main_prediction.setStyleSheet(f"color: {color}; margin-bottom: 20px;")
        
        # Add bars for top 5 predictions
        for i, idx in enumerate(top_indices):
            class_name = self.class_indices.get(idx, f"Class {idx}")
            prob = predictions[idx]
            
            # Create and add result bar
            result_bar = ResultBar(class_name, prob)
            self.results_layout.addWidget(result_bar)
        
        # Add a stretch at the end
        self.results_layout.addStretch()
        
        # Show results
        self.results_scroll.show()
    
    def reset_ui(self):
        # Hide image and results
        self.image_display.hide()
        self.results_header.hide()
        self.main_prediction.hide()
        self.results_scroll.hide()
        self.try_another_btn.hide()
        
        # Show drop area
        self.drop_area.show()
        
        # Clear image
        self.image_display.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitClassifierApp()
    window.show()
    sys.exit(app.exec_())
