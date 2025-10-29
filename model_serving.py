"""
Simple YOLO Image Classification Inference
Loads a pretrained YOLOv8 classification model and runs predictions on images
"""

from typing import List, Tuple
from io import BytesIO

from PIL import Image
import numpy as np

from ultralytics import YOLO


def load_model() -> YOLO:
	"""Load YOLOv8 classification model (pretrained)."""
	# Smallest pretrained classifier; good for CPU inference
	# Model file will be auto-downloaded on first run if not present
	return YOLO("yolov8n-cls.pt")


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
	"""Decode bytes into a PIL image in RGB mode."""
	image = Image.open(BytesIO(image_bytes)).convert("RGB")
	return image


def predict_topk(model: YOLO, image: Image.Image, top_k: int = 5) -> Tuple[List[str], List[float]]:
	"""Run classification and return top-k class names and probabilities."""
	# YOLO classification expects PIL Image or path
	results = model.predict(source=image, imgsz=224, device="cpu", verbose=False)
	res = results[0]
	probs = res.probs  # ultralytics.schemas.Probs
	if probs is None:
		return [], []

	# Get top-k indices sorted by probability
	k = min(top_k, len(probs.data))
	indices = np.argsort(-probs.data.cpu().numpy())[:k]
	class_names = [res.names[int(i)] for i in indices]
	scores = [float(probs.data[i]) for i in indices]
	return class_names, scores


if __name__ == "__main__":
	# Quick local smoke test using an online image path or local file
	# Example using PIL load from a local file:
	# img = Image.open("example.jpg").convert("RGB")
	# model = load_model()
	# labels, scores = predict_topk(model, img)
	# print(list(zip(labels, scores)))
	pass

