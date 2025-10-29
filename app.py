from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from io import BytesIO
import requests
from PIL import Image

from model_serving import load_model, read_image_from_bytes, predict_topk

app = FastAPI(title="YOLO Image Classifier API", version="1.0.0")


class UrlRequest(BaseModel):
	url: str = Field(..., description="Publicly reachable image URL")
	top_k: int = 5


class PredictResponse(BaseModel):
	labels: List[str]
	scores: List[float]


@app.on_event("startup")
async def startup_event():
	global model
	model = load_model()


@app.get("/")
async def root():
	return {"status": "ok", "message": "YOLO Image Classifier API"}


@app.post("/predict/url", response_model=PredictResponse)
async def predict_from_url(req: UrlRequest):
	try:
		resp = requests.get(req.url, timeout=10)
		resp.raise_for_status()
		img = Image.open(BytesIO(resp.content)).convert("RGB")
		labels, scores = predict_topk(model, img, top_k=req.top_k)
		return PredictResponse(labels=labels, scores=scores)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/file", response_model=PredictResponse)
async def predict_from_file(file: UploadFile = File(...), top_k: int = 5):
	try:
		image_bytes = await file.read()
		img = read_image_from_bytes(image_bytes)
		labels, scores = predict_topk(model, img, top_k=top_k)
		return PredictResponse(labels=labels, scores=scores)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
