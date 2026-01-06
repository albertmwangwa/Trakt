"""
OCR Engine Module

This module provides OCR functionality using TensorFlow, Tesseract, or EasyOCR.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict
import logging
import re


class OCREngine:
    """OCR engine for text detection and recognition."""
    
    def __init__(self, config: dict):
        """
        Initialize OCR engine.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config
        self.engine_type = config.get('engine', 'easyocr')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_text_length = config.get('min_text_length', 2)
        self.filter_patterns = config.get('filter_patterns', [])
        self.logger = logging.getLogger(__name__)
        
        # Initialize the selected OCR engine
        self.ocr_reader = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the OCR engine based on configuration."""
        try:
            if self.engine_type == 'tesseract':
                self._initialize_tesseract()
            elif self.engine_type == 'easyocr':
                self._initialize_easyocr()
            else:
                self.logger.error(f"Unknown OCR engine: {self.engine_type}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR engine: {e}")
    
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR."""
        import pytesseract
        self.logger.info("Initialized Tesseract OCR engine")
        self.ocr_reader = pytesseract
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR."""
        import easyocr
        
        easyocr_config = self.config.get('easyocr', {})
        languages = easyocr_config.get('languages', ['en'])
        gpu = easyocr_config.get('gpu', False)
        model_storage = easyocr_config.get('model_storage_directory', './models/easyocr')
        
        self.logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        self.ocr_reader = easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory=model_storage
        )
        self.logger.info("EasyOCR initialized successfully")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better OCR accuracy.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def detect_text_tesseract(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect text using Tesseract OCR.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection results
        """
        import pytesseract
        
        tesseract_config = self.config.get('tesseract', {})
        lang = tesseract_config.get('language', 'eng')
        oem = tesseract_config.get('oem', 3)
        psm = tesseract_config.get('psm', 6)
        
        custom_config = f'--oem {oem} --psm {psm}'
        
        # Get detailed data including bounding boxes
        data = pytesseract.image_to_data(
            frame,
            lang=lang,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if text and conf > self.confidence_threshold * 100:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                results.append({
                    'text': text,
                    'confidence': conf / 100.0,
                    'bbox': [x, y, x + w, y + h]
                })
        
        return results
    
    def detect_text_easyocr(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect text using EasyOCR.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection results
        """
        results_raw = self.ocr_reader.readtext(frame)
        
        results = []
        for detection in results_raw:
            bbox, text, confidence = detection
            
            if confidence > self.confidence_threshold:
                # Convert bbox to [x1, y1, x2, y2] format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                bbox_rect = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]
                
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox_rect
                })
        
        return results
    
    def detect_text(self, frame: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Detect text in frame.
        
        Args:
            frame: Input frame
            preprocess: Whether to preprocess the frame
            
        Returns:
            List of detection results with text, confidence, and bounding box
        """
        if preprocess:
            processed_frame = self.preprocess_frame(frame)
        else:
            processed_frame = frame
        
        try:
            if self.engine_type == 'tesseract':
                results = self.detect_text_tesseract(processed_frame)
            elif self.engine_type == 'easyocr':
                results = self.detect_text_easyocr(frame)  # EasyOCR works better with color
            else:
                return []
            
            # Filter results
            filtered_results = self._filter_results(results)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error during text detection: {e}")
            return []
    
    def _filter_results(self, results: List[Dict]) -> List[Dict]:
        """
        Filter OCR results based on configuration.
        
        Args:
            results: Raw OCR results
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            text = result['text']
            
            # Filter by minimum length
            if len(text) < self.min_text_length:
                continue
            
            # Apply pattern filtering if specified
            if self.filter_patterns:
                matches_pattern = False
                for pattern in self.filter_patterns:
                    if re.search(pattern, text):
                        matches_pattern = True
                        result['matched_pattern'] = pattern
                        break
                
                # Optionally, only keep results that match patterns
                # Uncomment to enable strict filtering:
                # if not matches_pattern:
                #     continue
            
            filtered.append(result)
        
        return filtered
    
    def annotate_frame(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Annotate frame with OCR results.
        
        Args:
            frame: Input frame
            results: OCR detection results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for result in results:
            text = result['text']
            confidence = result['confidence']
            bbox = result['bbox']
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(
                annotated,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )
            
            # Draw text and confidence
            label = f"{text} ({confidence:.2f})"
            cv2.putText(
                annotated,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return annotated


class TensorFlowOCRModel:
    """Custom TensorFlow model for OCR (template for custom models)."""
    
    def __init__(self, model_path: str, config: dict):
        """
        Initialize TensorFlow OCR model.
        
        Args:
            model_path: Path to saved TensorFlow model
            config: Model configuration
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # GPU configuration
        self._configure_gpu()
        
        # Load model if path exists
        if model_path:
            self._load_model()
    
    def _configure_gpu(self):
        """Configure TensorFlow GPU settings."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                gpu_memory_limit = self.config.get('gpu_memory_limit')
                allow_growth = self.config.get('allow_growth', True)
                
                for gpu in gpus:
                    if allow_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    if gpu_memory_limit:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=gpu_memory_limit
                            )]
                        )
                
                self.logger.info(f"Configured {len(gpus)} GPU(s) for TensorFlow")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")
    
    def _load_model(self):
        """Load TensorFlow model from file."""
        try:
            import os
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info(f"Loaded TensorFlow model from {self.model_path}")
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run prediction on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Model predictions
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return None
        
        # Preprocess frame to match model input shape
        input_shape = self.config.get('input_shape', [224, 224, 3])
        processed = cv2.resize(frame, (input_shape[1], input_shape[0]))
        processed = processed / 255.0  # Normalize
        processed = np.expand_dims(processed, axis=0)
        
        # Run prediction
        predictions = self.model.predict(processed, verbose=0)
        
        return predictions
