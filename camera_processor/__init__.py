"""
Camera Processor Module

This module provides computer vision capabilities for the EchoWear project,
including object detection, tracking, and analysis.
"""

from .object_detector import ObjectDetector, DetectionBackend

__all__ = ['ObjectDetector', 'DetectionBackend']
