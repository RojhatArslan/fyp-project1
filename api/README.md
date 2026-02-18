# Restaurant Demand Forecasting API

FastAPI backend for restaurant demand forecasting predictions.

## Setup

1. Install dependencies:
```bash
pip install -r api/requirements.txt
```

2. Train and save model:
```bash
python api/save_model.py
```

3. Start the server (choose one method):

**Option A - Simple script (recommended):**
```bash
python3 run.py
```
or
```bash
chmod +x run.sh
./run.sh
```

**Option B - Direct command:**
```bash
PYTHONPATH=. python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Option C - From api directory:**
```bash
cd api
python3 main.py
```

The API will automatically find a free port (8000, 8001, 8002...) and print the URL.
Open `http://localhost:<port>` in your browser to use the web interface.

## Endpoints

### POST /predict
Generate demand forecasts for a date range.

**Request:**
```json
{
  "start_date": "2026-02-01",
  "horizon_days": 7
}
```

### GET /health
Check API health and model status.

## Frontend

Open `api/static/index.html` in a browser or visit `http://localhost:8000`
