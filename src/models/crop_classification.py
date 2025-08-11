
"""
🌱 Crop Classification Module - Annam AI
========================================

This module implements crop classification using both traditional machine learning
and deep learning approaches with satellite imagery (EuroSAT dataset).

Key Features:
- CNN-based classification using TensorFlow/Keras
- Traditional ML using Random Forest and XGBoost
- Transfer learning with pre-trained models (ResNet50, EfficientNet)
- Data augmentation and preprocessing
- Model evaluation and visualization
- Export to TensorFlow Lite and ONNX

Author: Annam AI Team
License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm
import joblib
import json

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# PyTorch alternative
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import torch.nn.functional as F

import logging

# Set up logging
logger = logging.getLogger('AnnamAI.CropClassification')


class DataDownloader:
    """Handle downloading and preprocessing of crop classification datasets."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # EuroSAT dataset info
        self.eurosat_url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
        self.eurosat_classes = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]

    def download_file(self, url, filename):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
            desc=filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                progress_bar.update(size)

    def download_eurosat_dataset(self):
        """Download and extract EuroSAT dataset."""
        eurosat_zip = self.data_dir / "EuroSAT_RGB.zip"
        eurosat_dir = self.data_dir / "EuroSAT_RGB"

        if eurosat_dir.exists():
            logger.info("EuroSAT dataset already exists")
            return eurosat_dir

        logger.info("Downloading EuroSAT dataset...")
        self.download_file(self.eurosat_url, eurosat_zip)

        logger.info("Extracting EuroSAT dataset...")
        with zipfile.ZipFile(eurosat_zip, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        logger.info(f"EuroSAT dataset downloaded and extracted to {eurosat_dir}")
        return eurosat_dir

    def prepare_dataset_info(self, dataset_dir):
        """Prepare dataset information for training."""
        dataset_info = {
            'classes': [],
            'class_counts': {},
            'total_images': 0,
            'image_size': None
        }

        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

                if image_files:
                    dataset_info['classes'].append(class_name)
                    dataset_info['class_counts'][class_name] = len(image_files)
                    dataset_info['total_images'] += len(image_files)

                    # Get image size from first image
                    if dataset_info['image_size'] is None:
                        sample_img = Image.open(image_files[0])
                        dataset_info['image_size'] = sample_img.size

        return dataset_info


class CropClassificationCNN:
    """CNN-based crop classification model using TensorFlow/Keras."""

    def __init__(self, num_classes, img_size=(64, 64), use_pretrained=True):
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_pretrained = use_pretrained
        self.model = None
        self.history = None

    def create_model(self):
        """Create CNN model for crop classification."""
        if self.use_pretrained:
            # Transfer learning with ResNet50
            base_model = ResNet50(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )

            # Freeze base model layers
            base_model.trainable = False

            # Add custom classification head
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        else:
            # Custom CNN from scratch
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def create_data_generators(self, dataset_dir, batch_size=32, validation_split=0.2):
        """Create data generators for training and validation."""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_generator = val_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, val_generator

    def train(self, train_generator, val_generator, epochs=50, patience=10):
        """Train the CNN model."""
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7)
        ]

        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        if self.history is None:
            logger.error("No training history available. Train the model first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def predict_image(self, image_path, class_names):
        """Predict crop class for a single image."""
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        return class_names[predicted_class], confidence

    def export_to_tflite(self, filepath):
        """Export model to TensorFlow Lite format for mobile deployment."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(filepath, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"TensorFlow Lite model saved to {filepath}")


class CropClassificationML:
    """Traditional ML approach for crop classification using extracted features."""

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def extract_features(self, image_path):
        """Extract traditional computer vision features from images."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.resize(img, (64, 64))

        # Color features
        color_features = []
        for i in range(3):  # BGR channels
            channel = img[:, :, i]
            color_features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel)
            ])

        # Texture features using Local Binary Pattern (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Simple texture measures
        texture_features = [
            np.mean(gray),
            np.std(gray),
            np.var(gray)
        ]

        # Shape features (contours)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_features = [
            len(contours),
            np.sum([cv2.contourArea(c) for c in contours]),
            np.sum([cv2.arcLength(c, True) for c in contours])
        ]

        return color_features + texture_features + shape_features

    def prepare_dataset(self, dataset_dir):
        """Prepare feature dataset from images."""
        features = []
        labels = []

        logger.info("Extracting features from images...")

        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

                for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
                    feature_vector = self.extract_features(img_path)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(class_name)

        return np.array(features), np.array(labels)

    def train_models(self, X, y, test_size=0.2):
        """Train both Random Forest and XGBoost models."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train_scaled, y_train)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.xgb_model.fit(X_train_scaled, y_train)

        # Evaluate models
        rf_score = self.rf_model.score(X_test_scaled, y_test)
        xgb_score = self.xgb_model.score(X_test_scaled, y_test)

        logger.info(f"Random Forest accuracy: {rf_score:.4f}")
        logger.info(f"XGBoost accuracy: {xgb_score:.4f}")

        return X_test_scaled, y_test

    def plot_feature_importance(self):
        """Plot feature importance for Random Forest model."""
        if self.rf_model is None:
            logger.error("Random Forest model not trained yet.")
            return

        feature_names = [
            'B_mean', 'B_std', 'B_median', 'B_min', 'B_max',
            'G_mean', 'G_std', 'G_median', 'G_min', 'G_max',
            'R_mean', 'R_std', 'R_median', 'R_min', 'R_max',
            'Gray_mean', 'Gray_std', 'Gray_var',
            'Contour_count', 'Total_area', 'Total_perimeter'
        ]

        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Random Forest)")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

    def save_models(self, model_dir):
        """Save trained models."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.rf_model, model_dir / 'rf_crop_classifier.pkl')
        joblib.dump(self.xgb_model, model_dir / 'xgb_crop_classifier.pkl')
        joblib.dump(self.scaler, model_dir / 'feature_scaler.pkl')
        joblib.dump(self.label_encoder, model_dir / 'label_encoder.pkl')

        logger.info(f"ML models saved to {model_dir}")

    def predict_image(self, image_path, model_type='rf'):
        """Predict crop class for a single image using ML model."""
        features = self.extract_features(image_path)
        if features is None:
            return None, 0.0

        features_scaled = self.scaler.transform([features])

        if model_type == 'rf' and self.rf_model is not None:
            prediction = self.rf_model.predict(features_scaled)[0]
            confidence = np.max(self.rf_model.predict_proba(features_scaled))
        elif model_type == 'xgb' and self.xgb_model is not None:
            prediction = self.xgb_model.predict(features_scaled)[0]
            confidence = np.max(self.xgb_model.predict_proba(features_scaled))
        else:
            return None, 0.0

        class_name = self.label_encoder.inverse_transform([prediction])[0]
        return class_name, confidence


def main():
    """Main function to demonstrate crop classification pipeline."""
    logger.info("Starting Crop Classification Pipeline")

    # Initialize data downloader
    downloader = DataDownloader("data/crop_classification")

    # Download dataset
    dataset_dir = downloader.download_eurosat_dataset()
    dataset_info = downloader.prepare_dataset_info(dataset_dir)

    logger.info(f"Dataset info: {dataset_info}")

    # CNN approach
    logger.info("Training CNN model...")
    cnn_model = CropClassificationCNN(
        num_classes=len(dataset_info['classes']),
        img_size=(64, 64),
        use_pretrained=True
    )

    cnn_model.create_model()
    cnn_model.compile_model()

    train_gen, val_gen = cnn_model.create_data_generators(dataset_dir)
    cnn_model.train(train_gen, val_gen, epochs=10)  # Reduced for demo

    # Save CNN model
    cnn_model.save_model("data/models/cnn_crop_classifier.h5")
    cnn_model.export_to_tflite("data/models/cnn_crop_classifier.tflite")

    # Traditional ML approach
    logger.info("Training traditional ML models...")
    ml_model = CropClassificationML()

    X, y = ml_model.prepare_dataset(dataset_dir)
    X_test, y_test = ml_model.train_models(X, y)

    # Save ML models
    ml_model.save_models("data/models")

    logger.info("Crop Classification Pipeline completed!")


if __name__ == "__main__":
    main()
