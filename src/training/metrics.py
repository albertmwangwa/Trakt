"""
Metrics Module

Provides evaluation metrics for OCR models.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


class OCRMetrics:
    """Evaluation metrics for OCR models."""

    def __init__(self):
        """Initialize OCR metrics calculator."""
        self.logger = logging.getLogger(__name__)

    def character_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Calculate character-level accuracy.

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Character accuracy (0-1)
        """
        if not predictions or not ground_truths:
            return 0.0

        total_chars = 0
        correct_chars = 0

        for pred, gt in zip(predictions, ground_truths):
            min_len = min(len(pred), len(gt))
            max_len = max(len(pred), len(gt))

            total_chars += max_len

            for i in range(min_len):
                if pred[i] == gt[i]:
                    correct_chars += 1

        return correct_chars / total_chars if total_chars > 0 else 0.0

    def word_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Calculate word-level (exact match) accuracy.

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Word accuracy (0-1)
        """
        if not predictions or not ground_truths:
            return 0.0

        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if pred == gt
        )

        return correct / len(ground_truths)

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def character_error_rate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Calculate Character Error Rate (CER).

        CER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = reference length

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Character error rate (lower is better)
        """
        if not predictions or not ground_truths:
            return 1.0

        total_distance = 0
        total_length = 0

        for pred, gt in zip(predictions, ground_truths):
            total_distance += self.levenshtein_distance(pred, gt)
            total_length += len(gt)

        return total_distance / total_length if total_length > 0 else 1.0

    def word_error_rate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Calculate Word Error Rate (WER).

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Word error rate (lower is better)
        """
        if not predictions or not ground_truths:
            return 1.0

        total_distance = 0
        total_words = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_words = pred.split()
            gt_words = gt.split()

            total_distance += self.levenshtein_distance(
                " ".join(pred_words),
                " ".join(gt_words)
            )
            total_words += len(gt_words)

        return total_distance / total_words if total_words > 0 else 1.0

    def normalized_edit_distance(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Calculate Normalized Edit Distance (NED).

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Normalized edit distance (0-1, lower is better)
        """
        if not predictions or not ground_truths:
            return 1.0

        total_ned = 0

        for pred, gt in zip(predictions, ground_truths):
            distance = self.levenshtein_distance(pred, gt)
            max_len = max(len(pred), len(gt))
            if max_len > 0:
                total_ned += distance / max_len
            else:
                total_ned += 0

        return total_ned / len(ground_truths)

    def sequence_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
        case_sensitive: bool = True
    ) -> float:
        """
        Calculate sequence accuracy (exact match).

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings
            case_sensitive: Whether to consider case

        Returns:
            Sequence accuracy (0-1)
        """
        if not predictions or not ground_truths:
            return 0.0

        if not case_sensitive:
            predictions = [p.lower() for p in predictions]
            ground_truths = [g.lower() for g in ground_truths]

        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if pred == gt
        )

        return correct / len(ground_truths)

    def precision_recall_f1(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate character-level precision, recall, and F1 score.

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not predictions or not ground_truths:
            return 0.0, 0.0, 0.0

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_chars = list(pred)
            gt_chars = list(gt)

            # Count character occurrences
            pred_counts = {}
            gt_counts = {}

            for c in pred_chars:
                pred_counts[c] = pred_counts.get(c, 0) + 1

            for c in gt_chars:
                gt_counts[c] = gt_counts.get(c, 0) + 1

            # Calculate TP, FP, FN
            all_chars = set(pred_counts.keys()) | set(gt_counts.keys())

            for c in all_chars:
                pred_count = pred_counts.get(c, 0)
                gt_count = gt_counts.get(c, 0)

                tp = min(pred_count, gt_count)
                true_positives += tp
                false_positives += pred_count - tp
                false_negatives += gt_count - tp

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics for evaluation.

        Args:
            predictions: List of predicted text strings
            ground_truths: List of ground truth text strings

        Returns:
            Dictionary of all metrics
        """
        precision, recall, f1 = self.precision_recall_f1(predictions, ground_truths)

        return {
            "character_accuracy": self.character_accuracy(predictions, ground_truths),
            "word_accuracy": self.word_accuracy(predictions, ground_truths),
            "character_error_rate": self.character_error_rate(predictions, ground_truths),
            "word_error_rate": self.word_error_rate(predictions, ground_truths),
            "normalized_edit_distance": self.normalized_edit_distance(
                predictions, ground_truths
            ),
            "sequence_accuracy": self.sequence_accuracy(predictions, ground_truths),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics dictionary as a readable string.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Formatted string
        """
        lines = [
            "=" * 50,
            "OCR Evaluation Metrics",
            "=" * 50,
            f"Character Accuracy:      {metrics.get('character_accuracy', 0):.4f}",
            f"Word Accuracy:           {metrics.get('word_accuracy', 0):.4f}",
            f"Character Error Rate:    {metrics.get('character_error_rate', 0):.4f}",
            f"Word Error Rate:         {metrics.get('word_error_rate', 0):.4f}",
            f"Normalized Edit Dist:    {metrics.get('normalized_edit_distance', 0):.4f}",
            f"Sequence Accuracy:       {metrics.get('sequence_accuracy', 0):.4f}",
            "-" * 50,
            f"Precision:               {metrics.get('precision', 0):.4f}",
            f"Recall:                  {metrics.get('recall', 0):.4f}",
            f"F1 Score:                {metrics.get('f1_score', 0):.4f}",
            "=" * 50,
        ]

        return "\n".join(lines)
