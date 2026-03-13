# ARCDD AI — Disease Detection Flask API

EfficientNetB2 model API for Rice and Pulse Crop Disease Detection.

## Endpoints
- `GET /` — Health check
- `POST /predict` — Upload image, returns disease prediction

## Local Development
```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Render
1. Push this folder to GitHub
2. Go to render.com → New Web Service
3. Connect repo → Deploy
