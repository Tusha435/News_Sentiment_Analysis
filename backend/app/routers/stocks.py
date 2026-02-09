"""Stock data API endpoints."""

from fastapi import APIRouter, Query, HTTPException

from backend.app.services.alpha_vantage import av_client
from backend.app.services.technical import technical_indicators

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])


@router.get("/search")
async def search_stocks(q: str = Query(..., min_length=1, description="Search query")):
    """Search for stock symbols by keyword."""
    try:
        results = av_client.search_symbol(q)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/prices")
async def get_stock_prices(
    symbol: str,
    days: int = Query(365, ge=1, le=7300, description="Number of days of history"),
):
    """Get historical stock prices with technical indicators."""
    try:
        from datetime import datetime, timedelta
        price_df = av_client.get_daily_prices(symbol.upper())

        if price_df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")

        cutoff = datetime.utcnow() - timedelta(days=days)
        price_df = price_df[price_df["date"] >= cutoff].reset_index(drop=True)

        # Add technical indicators
        tech_df = technical_indicators.compute_all(price_df)

        records = []
        for _, row in tech_df.iterrows():
            records.append({
                "date": row["date"].isoformat(),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
                "sma_20": round(float(row.get("sma_20", 0)), 2),
                "sma_50": round(float(row.get("sma_50", 0)), 2),
                "ema_12": round(float(row.get("ema_12", 0)), 2),
                "rsi_14": round(float(row.get("rsi_14", 50)), 2),
                "macd": round(float(row.get("macd", 0)), 4),
                "macd_signal": round(float(row.get("macd_signal", 0)), 4),
                "bb_upper": round(float(row.get("bb_upper", 0)), 2),
                "bb_lower": round(float(row.get("bb_lower", 0)), 2),
                "bb_middle": round(float(row.get("bb_middle", 0)), 2),
                "atr_14": round(float(row.get("atr_14", 0)), 4),
                "daily_return": round(float(row.get("daily_return", 0)), 6),
                "volatility_20": round(float(row.get("volatility_20", 0)), 6),
            })

        return {
            "symbol": symbol.upper(),
            "count": len(records),
            "prices": records,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/overview")
async def get_company_overview(symbol: str):
    """Get company profile and fundamental data."""
    try:
        overview = av_client.get_company_overview(symbol.upper())
        if not overview or "Symbol" not in overview:
            raise HTTPException(status_code=404, detail=f"No overview data for {symbol}")
        return overview
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    """Get company financial statements."""
    try:
        income = av_client.get_income_statement(symbol.upper())
        balance = av_client.get_balance_sheet(symbol.upper())
        cashflow = av_client.get_cash_flow(symbol.upper())
        earnings = av_client.get_earnings(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
            "earnings": earnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
