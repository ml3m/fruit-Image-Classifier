import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QProgressBar, QFrame, QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
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
    image_dropped = pyqtSignal(str)  # Signal to notify when an image is dropped
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n🍎 Drop your fruit image here 🍌\n\nor click to browse files\n\n")
        self.setFont(QFont("Arial", 14))
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 15px;
                background-color: #f8f8f8;
                padding: 40px;
                color: #666;
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
                    border-radius: 15px;
                    background-color: #e8f6ff;
                    padding: 40px;
                    color: #3498db;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 15px;
                background-color: #f8f8f8;
                padding: 40px;
                color: #666;
            }
        """)
            
    def dropEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 15px;
                background-color: #f8f8f8;
                padding: 40px;
                color: #666;
            }
        """)
        
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_dropped.emit(file_path)  # Emit signal with file path
                    break
                    
    def mousePressEvent(self, event):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)'
        )
        if file_path:
            self.image_dropped.emit(file_path)  # Emit signal with file path


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
                border-radius: 8px;
                padding: 3px;
                border: 1px solid #e0e0e0;
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


class FruitClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Classifier")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QWidget {
                font-family: Arial;
            }
            QPushButton {
                height: 36px;
                font-weight: bold;
            }
        """)
        
        self.loadModel()
        self.initUI()
        
    def loadModel(self):
        # Load model (try both filenames)
        if os.path.exists('best_model.h5'):
            self.model_path = 'best_model.h5'
        elif os.path.exists('fruit_model.h5'):
            self.model_path = 'fruit_model.h5'
        else:
            self.model = None
            return
            
        self.model = load_model(self.model_path)
        
        # Load class indices
        self.class_indices = {}
        try:
            with open('class_indices.txt', 'r') as f:
                for line in f:
                    fruit, idx = line.strip().split(': ')
                    self.class_indices[int(idx)] = fruit
        except Exception as e:
            print(f"Error loading class indices: {e}")
            self.class_indices = {}
        
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Create a card-like container for the content
        content_card = QFrame()
        content_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        card_layout = QVBoxLayout(content_card)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(25, 25, 25, 25)
        
        # Add shadow effect to the card
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        content_card.setGraphicsEffect(shadow)
        
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
        card_layout.addWidget(header)
        
        # Subtitle
        subtitle = QLabel("Drop or select an image to identify the fruit")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        card_layout.addWidget(subtitle)
        
        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        
        # Left panel - Drop area and image display
        left_panel = QVBoxLayout()
        
        # Drop area for images
        self.drop_area = DropArea(self)
        self.drop_area.image_dropped.connect(self.load_image)  # Connect signal to method
        left_panel.addWidget(self.drop_area)
        
        # Image display (initially hidden)
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
                padding: 10px;
            }
        """)
        image_layout = QVBoxLayout(self.image_frame)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 300)
        self.image_display.setMaximumSize(400, 400)
        self.image_display.setStyleSheet("""
            QLabel {
                border-radius: 10px;
            }
        """)
        self.image_display.setScaledContents(True)
        image_layout.addWidget(self.image_display)
        
        self.image_frame.hide()
        left_panel.addWidget(self.image_frame)
        
        # "Try Another" button (initially hidden)
        self.try_another_btn = QPushButton("Try Another Image")
        self.try_another_btn.setFont(QFont("Arial", 11))
        self.try_another_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                padding: 12px;
                margin-top: 15px;
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
        
        # Results section in a card
        self.results_card = QFrame()
        self.results_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
                padding: 15px;
            }
        """)
        results_card_layout = QVBoxLayout(self.results_card)
        
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
        results_card_layout.addWidget(self.results_header)
        
        # Main prediction
        self.main_prediction = QLabel()
        self.main_prediction.setFont(QFont("Arial", 14))
        self.main_prediction.setStyleSheet("color: #27ae60; margin-bottom: 20px;")
        self.main_prediction.setWordWrap(True)
        results_card_layout.addWidget(self.main_prediction)
        
        # Description label
        self.description_label = QLabel("Here are the top matches from our fruit database:")
        self.description_label.setFont(QFont("Arial", 11))
        self.description_label.setStyleSheet("color: #555; margin-bottom: 10px;")
        results_card_layout.addWidget(self.description_label)
        
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
        self.results_layout.setSpacing(10)
        self.results_layout.setContentsMargins(5, 5, 5, 5)
        
        # Scroll area for results
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setWidget(self.results_container)
        self.results_scroll.setFrameShape(QFrame.NoFrame)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_scroll.setMinimumWidth(400)
        self.results_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        results_card_layout.addWidget(self.results_scroll)
        
        self.results_card.hide()
        right_panel.addWidget(self.results_card)
        
        # Status message for no model
        if self.model is None:
            self.status_label = QLabel("⚠️ No model found. Please place 'best_model.h5' or 'fruit_model.h5' in the same directory.")
            self.status_label.setFont(QFont("Arial", 12))
            self.status_label.setStyleSheet("color: #e74c3c; margin: 20px;")
            self.status_label.setWordWrap(True)
            self.status_label.setAlignment(Qt.AlignCenter)
            right_panel.addWidget(self.status_label)
        
        content_layout.addLayout(right_panel)
        card_layout.addLayout(content_layout)
        
        # Bottom credits
        credits = QLabel("Powered by TensorFlow and ResNet50")
        credits.setFont(QFont("Arial", 9))
        credits.setAlignment(Qt.AlignCenter)
        credits.setStyleSheet("color: #95a5a6; margin-top: 10px;")
        card_layout.addWidget(credits)
        
        # Add the card to the main layout
        main_layout.addWidget(content_card)
        
        self.setCentralWidget(main_widget)
        
    def load_image(self, file_path):
        if self.model is None:
            return
            
        # Hide drop area
        self.drop_area.hide()
        
        # Display the image
        pixmap = QPixmap(file_path)
        self.image_display.setPixmap(pixmap)
        self.image_frame.show()
        
        # Add shadow effect to image
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 3)
        self.image_display.setGraphicsEffect(shadow)
        
        # Show try again button
        self.try_another_btn.show()
        
        # Show results card with loading state
        self.results_card.show()
        self.results_header.setText("Processing...")
        self.main_prediction.setText("Analyzing image...")
        self.main_prediction.setStyleSheet("color: #3498db; margin-bottom: 20px;")
        self.description_label.setText("Please wait while we identify the fruit...")
        
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Add a "Processing" indicator
        processing_label = QLabel("Analyzing image... This may take a moment.")
        processing_label.setAlignment(Qt.AlignCenter)
        processing_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
        self.results_layout.addWidget(processing_label)
        
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
        
        # Update description
        self.description_label.setText("Here are the top matches from our fruit database:")
        
        # Add bars for top 5 predictions
        for i, idx in enumerate(top_indices):
            class_name = self.class_indices.get(idx, f"Class {idx}")
            prob = predictions[idx]
            
            # Create and add result bar
            result_bar = ResultBar(class_name, prob)
            self.results_layout.addWidget(result_bar)
            
            # Add separator except after last item
            if i < len(top_indices) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setStyleSheet("background-color: #e0e0e0;")
                separator.setMaximumHeight(1)
                self.results_layout.addWidget(separator)
        
        # Add a stretch at the end
        self.results_layout.addStretch()
    
    def reset_ui(self):
        # Hide image and results
        self.image_frame.hide()
        self.results_card.hide()
        self.try_another_btn.hide()
        
        # Show drop area
        self.drop_area.show()
        
        # Clear image
        self.image_display.clear()
        self.image_display.setGraphicsEffect(None)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitClassifierApp()
    window.show()
    sys.exit(app.exec_())
