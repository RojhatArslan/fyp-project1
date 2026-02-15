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

3. Start the server:
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

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
