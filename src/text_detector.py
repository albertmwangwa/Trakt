"""
Text Detection Module

This module provides text detection capabilities using the EAST text detector
and other detection methods to identify text regions in images before OCR.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np


class EASTTextDetector:
    """EAST (Efficient and Accurate Scene Text) detector for text region detection."""

    def __init__(self, model_path: str = None, config: dict = None):
        """
        Initialize EAST text detector.

        Args:
            model_path: Path to EAST model file (.pb format)
            config: Configuration dictionary for detector settings
        """
        self.model_path = model_path
        self.config = config or {}
        self.net = None
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.nms_threshold = self.config.get("nms_threshold", 0.4)
        self.input_width = self.config.get("input_width", 320)
        self.input_height = self.config.get("input_height", 320)

        # Initialize model if path provided
        if model_path:
            self._load_model()

    def _load_model(self):
        """Load EAST text detection model."""
        try:
            import os

            if not os.path.exists(self.model_path):
                self.logger.warning(
                    f"EAST model not found at {self.model_path}. "
                    "Please download from: "
                    "https://github.com/oyyd/frozen-east-text-detection.pb/raw/master/frozen_east_text_detection.pb"
                )
                return

            self.net = cv2.dnn.readNet(self.model_path)
            self.logger.info(f"Loaded EAST model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load EAST model: {e}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.net is None:
            self.logger.warning("EAST model not loaded, cannot detect text")
            return []

        orig = image.copy()
        (H, W) = image.shape[:2]

        # Calculate aspect ratio for resizing
        (newW, newH) = (self.input_width, self.input_height)
        rW = W / float(newW)
        rH = H / float(newH)

        # Resize image for the model
        image = cv2.resize(image, (newW, newH))

        # Prepare blob for the network
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )

        # Forward pass through the network
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # Decode predictions
        boxes, confidences = self._decode_predictions(scores, geometry, self.confidence_threshold)

        # Apply Non-Maximum Suppression
        boxes = self._apply_nms(boxes, confidences, self.nms_threshold)

        # Scale boxes back to original image size
        scaled_boxes = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            scaled_boxes.append((startX, startY, endX, endY))

        return scaled_boxes

    def _decode_predictions(
        self, scores: np.ndarray, geometry: np.ndarray, min_confidence: float
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Decode predictions from EAST model output.

        Args:
            scores: Score predictions from model
            geometry: Geometry predictions from model
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (bounding boxes, confidence scores)
        """
        (numRows, numCols) = scores.shape[2:4]
        boxes = []
        confidences = []

        # Loop over rows and columns
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # If score is below threshold, skip
                if scoresData[x] < min_confidence:
                    continue

                # Calculate offset factor
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # Extract rotation angle and compute sin/cos
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # Use geometry volume to derive bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # Compute start and end coordinates
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                boxes.append((startX, startY, endX, endY))
                confidences.append(float(scoresData[x]))

        return boxes, confidences

    def _apply_nms(
        self, boxes: List[Tuple[int, int, int, int]], confidences: List[float], threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.

        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores for each box
            threshold: IoU threshold for NMS

        Returns:
            Filtered list of bounding boxes
        """
        if len(boxes) == 0:
            return []

        # Convert to numpy array
        boxes_array = np.array(boxes)

        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes_array.tolist(),
            confidences,
            self.confidence_threshold,
            threshold,
        )

        # Extract filtered boxes
        if len(indices) > 0:
            indices = indices.flatten()
            return [boxes[i] for i in indices]

        return []


class TextRegionPreprocessor:
    """Preprocessor for enhancing text regions before OCR."""

    def __init__(self, config: dict = None):
        """
        Initialize text region preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image by detecting and correcting rotation.

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply binary threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find all non-zero points
        coords = np.column_stack(np.where(thresh > 0))

        # Calculate minimum area rectangle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]

            # Correct angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Only deskew if angle is significant
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return rotated

        return image

    def correct_perspective(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Correct perspective distortion in image.

        Args:
            image: Input image
            contour: Contour defining the text region

        Returns:
            Perspective-corrected image
        """
        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Order points: top-left, top-right, bottom-right, bottom-left
        box = self._order_points(box)

        # Calculate width and height of new image
        (tl, tr, br, bl) = box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Define destination points
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(box.astype("float32"), dst)

        # Apply perspective transformation
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts: Array of 4 points

        Returns:
            Ordered points
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference to find corners
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image using morphological operations.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Apply bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(closing, 9, 75, 75)

        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return denoised

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize illumination in image.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back
        if len(image.shape) == 3:
            lab = cv2.merge([l, a, b])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            normalized = l

        return normalized

    def preprocess_text_region(self, image: np.ndarray, apply_all: bool = False) -> np.ndarray:
        """
        Apply comprehensive preprocessing to text region.

        Args:
            image: Input image (text region)
            apply_all: Apply all preprocessing steps (default: False, only essential)

        Returns:
            Preprocessed image
        """
        processed = image.copy()

        # Always apply these
        processed = self.normalize_illumination(processed)
        processed = self.deskew(processed)

        # Optional preprocessing
        if apply_all:
            processed = self.enhance_contrast(processed)
            processed = self.remove_noise(processed)

        return processed
