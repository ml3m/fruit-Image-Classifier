import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Lighter than ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

# Enable memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU available: {physical_devices}")
else:
    print("No GPU detected. Using CPU.")

# Enable mixed precision for faster training on compatible GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set paths
dataset_path = 'data/fruits/Fruit-Images-Dataset-master/Training'
validation_path = 'data/fruits/Fruit-Images-Dataset-master/Test'
model_save_path = 'fruit_model.h5'
checkpoint_path = 'checkpoints/model-epoch-{epoch:02d}.h5'  # Full model per epoch
best_model_path = 'best_model.h5'

# Create checkpoint directory if it doesn't exist
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Parameters
img_width, img_height = 224, 224
batch_size = 64  # Increased batch size for faster training
epochs = 10
buffer_size = 1000  # For dataset shuffling

# Function to create a dataset from directory
def create_dataset(directory, is_training=True):
    # Get all image files
    image_files = []
    labels = []
    class_names = sorted(os.listdir(directory))
    class_dict = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(class_dir, img_file))
                    labels.append(class_dict[class_name])
    
    # Create a dataset
    def preprocess_image(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_width, img_height])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, tf.one_hot(label, len(class_dict))
    
    # Create dataset from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    
    # Shuffle only training data
    if is_training:
        dataset = dataset.shuffle(buffer_size)
    
    # Preprocess, batch, and prefetch
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Data augmentation for training
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, label
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_dict

# Create training and validation datasets
print("Creating datasets...")
train_dataset, class_indices = create_dataset(dataset_path, is_training=True)
validation_dataset, _ = create_dataset(validation_path, is_training=False)

num_classes = len(class_indices)
print(f"Found {num_classes} classes")

# Save the class indices mapping
with open('class_indices.txt', 'w') as f:
    for fruit, index in class_indices.items():
        f.write(f"{fruit}: {index}\n")

# Calculate steps
train_steps = tf.data.experimental.cardinality(train_dataset).numpy()
val_steps = tf.data.experimental.cardinality(validation_dataset).numpy()
print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

# Load MobileNetV2 model (faster than ResNet50)
base_model = MobileNetV2(weights='imagenet', 
                        include_top=False, 
                        input_shape=(img_width, img_height, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Smaller dense layer
x = Dropout(0.3)(x)  # Add dropout to prevent overfitting
predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Ensure float32 output

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Callbacks
callbacks = [
    # Save complete model after each epoch
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,  # Save full model
        verbose=1,
        save_freq='epoch'
    ),
    # Save the best model based on validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1,
        save_weights_only=False  # Save full model
    ),
    # Learning rate reduction on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001
    ),
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        update_freq='epoch'
    )
]

# Train the model (Phase 1)
print("\nStarting initial training phase...")
start_time = time.time()
history1 = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=callbacks
)
phase1_time = time.time() - start_time
print(f"Phase 1 completed in {phase1_time:.2f} seconds")

# Fine-tune: Unfreeze some layers for better accuracy
print("\nUnfreezing top layers for fine-tuning...")
for layer in base_model.layers[-20:]:  # Unfreeze more layers
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with fine-tuning (Phase 2)
print("\nStarting fine-tuning phase...")
start_time = time.time()
history2 = model.fit(
    train_dataset,
    epochs=5,  # Complete the 10 total epochs (5 + 5)
    validation_data=validation_dataset,
    callbacks=callbacks
)
phase2_time = time.time() - start_time
print(f"Phase 2 completed in {phase2_time:.2f} seconds")

# Save the final model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Combine histories for plotting
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# Plot full training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(combined_history['accuracy'])
plt.plot(combined_history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.axvline(x=4.5, color='r', linestyle='--')  # Mark where fine-tuning starts

plt.subplot(1, 2, 2)
plt.plot(combined_history['loss'])
plt.plot(combined_history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.axvline(x=4.5, color='r', linestyle='--')  # Mark where fine-tuning starts
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Print total training time
total_time = phase1_time + phase2_time
print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Evaluate the model
print("\nEvaluating final model...")
evaluation = model.evaluate(validation_dataset)
print(f"Final validation loss: {evaluation[0]:.4f}")
print(f"Final validation accuracy: {evaluation[1]:.4f}")
