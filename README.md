# YOLO Image Classifier API

A simple image classification API using Ultralytics YOLOv8 (pretrained `yolov8n-cls.pt`).

## Files

- `app.py` - FastAPI app exposing `/predict/file` and `/predict/url`
- `model_serving.py` - Loads YOLO model and runs top-k classification
- `requirements.txt` - Dependencies (Ultralytics, FastAPI, etc.)
- `Dockerfile` - Container image for local and DigitalOcean deployments
- `.dockerignore` - Keeps image lean

## Quick Start (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Open docs at: `http://localhost:8000/docs`

### Example Requests

- Predict from URL:
  ```bash
  curl -X POST \
    http://localhost:8000/predict/url \
    -H "Content-Type: application/json" \
    -d '{
      "url": "https://ultralytics.com/images/bus.jpg",
      "top_k": 5
    }'
  ```

- Predict from file:
  ```bash
  curl -X POST http://localhost:8000/predict/file \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.jpg" \
    -F "top_k=5"
  ```

Response format:
```json
{
  "labels": ["school_bus", "bus", "van", "..."],
  "scores": [0.98, 0.01, 0.003]
}
```

## Docker

Build and run locally:
```bash
# From project root
docker build -t yolo-image-classifier:latest .
docker run -p 8000:8000 --name yolo-cls yolo-image-classifier:latest
```

Test:
```bash
curl -X POST \
  http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://ultralytics.com/images/bus.jpg", "top_k": 5}'
```

## Deploy to DigitalOcean (App Platform)

1. Push this project to a Git repo (GitHub/GitLab/Bitbucket).
2. In DigitalOcean, create a new App and select your repo.
3. Choose Dockerfile as the build method (Dockerfile is in project root).
4. Set the service port to 8000.
5. Optional environment:
   - `PYTHONDONTWRITEBYTECODE=1`
   - `PYTHONUNBUFFERED=1`
6. Deploy. The build stage warms up YOLO weights to speed up first request.
7. Your API will be available at your app URL (e.g., `https://<your-app>.ondigitalocean.app`).

### Deploy to a Droplet (optional)
```bash
# SSH into droplet, install Docker, then:
git clone <your-repo>.git
cd model-testing
sudo docker build -t yolo-image-classifier:latest .
sudo docker run -d -p 80:8000 --name yolo-cls yolo-image-classifier:latest
```

- Your API will be available on port 80 of the droplet.
