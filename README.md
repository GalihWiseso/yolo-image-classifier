# Linear Regression API

A simple regression API using scikit-learn LinearRegression trained on synthetic data.

## Files

- `train_model.py` - Trains LinearRegression and saves artifacts
- `model_serving.py` - Loads model and performs predictions
- `app.py` - FastAPI service exposing `/predict`
- `requirements.txt` - Minimal dependencies
- `Dockerfile` - Lightweight container for deployment
- `.dockerignore` - Keeps image lean

## Quick Start (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (creates `models/linear_regression.pkl`):
   ```bash
   python train_model.py
   ```

3. Run the API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. Open docs at: `http://localhost:8000/docs`

### Example Request

POST `/predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "feature_0": 0.1,
        "feature_1": -0.2,
        "feature_2": 1.5,
        "feature_3": 0.0,
        "feature_4": 2.2,
        "feature_5": -1.0,
        "feature_6": 0.3,
        "feature_7": 0.7,
        "feature_8": -0.4,
        "feature_9": 1.1
      }
    ]
  }'
```

Response:
```json
{
  "predictions": [123.4567]
}
```

## Docker

Build and run locally:
```bash
# From project root
docker build -t linear-regression-api:latest .
docker run -p 8000:8000 --name lr-api linear-regression-api:latest
```

Test:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature_0": 0.1, "feature_1": -0.2, "feature_2": 1.5, "feature_3": 0.0, "feature_4": 2.2, "feature_5": -1.0, "feature_6": 0.3, "feature_7": 0.7, "feature_8": -0.4, "feature_9": 1.1}]}'
```

## Deploy to DigitalOcean (App Platform)

1. Push this project to GitHub.
2. In DigitalOcean, create a new App and select your repo.
3. Choose Dockerfile as the build method.
4. Set the service port to 8000.
5. Deploy. Your API will be available at your app URL.
