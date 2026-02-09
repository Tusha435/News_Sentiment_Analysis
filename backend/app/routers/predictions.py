"""Prediction API endpoints."""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from backend.app.services.prediction import prediction_engine

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


class TrainRequest(BaseModel):
    symbol: str
    model_type: str = "hybrid"
    days: int = 730


@router.get("/{symbol}")
async def get_predictions(
    symbol: str,
    model_type: str = Query("hybrid", description="Model type: hybrid, xgboost, random_forest, lstm"),
):
    """Get stock price predictions for a symbol."""
    try:
        result = prediction_engine.predict(symbol.upper(), model_type)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a prediction model for a stock symbol."""
    try:
        result = prediction_engine.train_model(
            request.symbol.upper(),
            request.model_type,
            request.days,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"status": "success", "metrics": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/dashboard")
async def get_dashboard(
    symbol: str,
    days: int = Query(365, ge=30, le=3650),
):
    """Get full dashboard data for a symbol.

    Returns price history, predictions, sentiment timeline,
    reputation scoring, technical indicators, and news.
    """
    try:
        result = prediction_engine.get_dashboard_data(symbol.upper(), days)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
