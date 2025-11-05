from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

from model_serving import load_model, make_dataframe, predict

app = FastAPI(title="Linear Regression API", version="1.0.0")


class Record(BaseModel):
	# Expect 10 features named feature_0..feature_9
	feature_0: float = Field(...)
	feature_1: float = Field(...)
	feature_2: float = Field(...)
	feature_3: float = Field(...)
	feature_4: float = Field(...)
	feature_5: float = Field(...)
	feature_6: float = Field(...)
	feature_7: float = Field(...)
	feature_8: float = Field(...)
	feature_9: float = Field(...)


class PredictRequest(BaseModel):
	records: List[Record]


class PredictResponse(BaseModel):
	predictions: List[float]


@app.on_event("startup")
async def startup_event():
	global model, feature_names
	try:
		model, feature_names = load_model()
	except FileNotFoundError:
		# Attempt to train if artifacts missing
		import subprocess
		subprocess.run(["python3", "train_model.py"], check=True)
		model, feature_names = load_model()


@app.get("/")
async def root():
	return {"status": "ok", "message": "Linear Regression API"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
	try:
		df = make_dataframe([r.model_dump() for r in req.records], feature_names)
		preds = predict(model, df)
		preds_list = preds.tolist()
		return PredictResponse(predictions=preds_list)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
