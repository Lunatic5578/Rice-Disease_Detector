import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
train_path = 'D:\FY_Project/split_dataset/train'
test_path = 'D:\FY_Project/split_dataset/test'
validation_path = 'D:\FY_Project/split_dataset/validation'

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

# Train Data
train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Validation Data
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_data = val_datagen.flow_from_directory(
    validation_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Load Pre-trained Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained layers

# Model Definition
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile Model
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# Debugging prints
print(f"Number of training samples: {train_data.samples}")
print(f"Number of validation samples: {val_data.samples}")
print(f"Class weights dictionary: {class_weights_dict}")

# Train the Model (Initial Training)
history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# Fine-Tuning
# Unfreeze some layers in the base model
for layer in base_model.layers[-50:]:  # Unfreeze the last 50 layers
    layer.trainable = True

# Recompile Model for Fine-Tuning
model.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tune the Model
history_fine = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the Model
model.save("Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras")

# Evaluation
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2f}, Validation Loss: {val_loss:.2f}")

# Test Data Evaluation
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}")

# True labels from the test dataset
y_true = test_data.classes  # Ground truth labels

# Predicted labels (take the index of the max probability)
y_pred = np.argmax(model.predict(test_data), axis=1)

# Class labels (mapping of indices to class names)
class_labels = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
