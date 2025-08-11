"""
Unit tests for crop classification module.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.crop_classification import CropClassificationCNN, CropClassificationML

class TestCropClassification(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 5
        self.img_size = (64, 64)  # Smaller size for testing

    def test_cnn_model_creation(self):
        """Test CNN model creation."""
        classifier = CropClassificationCNN(
            num_classes=self.num_classes,
            img_size=self.img_size,
            use_pretrained=False  # Faster for testing
        )

        model = classifier.create_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 9)  # Expected number of layers

    def test_cnn_model_compilation(self):
        """Test CNN model compilation."""
        classifier = CropClassificationCNN(
            num_classes=self.num_classes,
            img_size=self.img_size,
            use_pretrained=False
        )

        classifier.create_model()
        classifier.compile_model()

        self.assertIsNotNone(classifier.model)
        self.assertEqual(classifier.model.optimizer.__class__.__name__, 'Adam')

    def test_ml_model_initialization(self):
        """Test traditional ML model initialization."""
        classifier = CropClassificationML()

        self.assertIsNone(classifier.rf_model)
        self.assertIsNone(classifier.xgb_model)
        self.assertIsNotNone(classifier.scaler)
        self.assertIsNotNone(classifier.label_encoder)

    def test_feature_extraction(self):
        """Test feature extraction from synthetic image."""
        classifier = CropClassificationML()

        # Create synthetic image
        synthetic_image_path = "test_image.jpg"
        from PIL import Image
        img = Image.new('RGB', (64, 64), color=(0, 128, 0))
        img.save(synthetic_image_path)

        features = classifier.extract_features(synthetic_image_path)

        # Clean up
        os.remove(synthetic_image_path)

        self.assertIsNotNone(features)
        self.assertEqual(len(features), 21)  # Expected number of features

if __name__ == '__main__':
    unittest.main()
