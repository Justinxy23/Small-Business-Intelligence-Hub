"""
Small Business Intelligence Hub - API Endpoints
Author: Justin Christopher Weaver
Description: FastAPI endpoints for business intelligence services
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, date
import pandas as pd
from sales_analytics import SalesAnalytics
import asyncio
import redis
import json
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI(
    title="Small Business Intelligence Hub API",
    description="Real-time analytics and insights for small businesses",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Pydantic models for request/response
class SalesTransaction(BaseModel):
    date: date
    product_id: str
    customer_id: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)
    total_amount: float
    category: str

class MetricsResponse(BaseModel):
    total_revenue: float
    total_transactions: int
    average_order_value: float
    unique_customers: int
    unique_products: int
    revenue_per_customer: float
    revenue_growth_30d: Optional[float] = None
    timestamp: datetime

class CustomerSegment(BaseModel):
    customer_id: str
    recency: int
    frequency: int
    monetary: float
    segment_name: str
    clv_prediction: float

class InsightResponse(BaseModel):
    insight_type: str
    message: str
    priority: str  # 'high', 'medium', 'low'
    action_required: bool

class DashboardData(BaseModel):
    metrics: MetricsResponse
    top_products: List[Dict]
    customer_segments: Dict[str, int]
    recent_anomalies: List[Dict]
    insights: List[InsightResponse]

# Dependency for data loading
@lru_cache()
def get_analytics_engine():
    """Load and cache analytics engine"""
    # In production, this would load from database
    # For demo, using synthetic data
    data = pd.read_csv('sales_data.csv')
    return SalesAnalytics(data)

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Small Business Intelligence Hub",
        "version": "1.0.0"
    }

@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_business_metrics(
    start_date: Optional[date] = Query(None, description="Start date for metrics calculation"),
    end_date: Optional[date] = Query(None, description="End date for metrics calculation"),
    use_cache: bool = Query(True, description="Use cached results if available")
):
    """
    Get key business metrics with optional date filtering
    """
    cache_key = f"metrics:{start_date}:{end_date}"
    
    # Check cache first
    if use_cache:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    
    try:
        analytics = get_analytics_engine()
        
        # Filter data by date if provided
        if start_date or end_date:
            filtered_data = analytics.data.copy()
            if start_date:
                filtered_data = filtered_data[filtered_data['date'] >= pd.to_datetime(start_date)]
            if end_date:
                filtered_data = filtered_data[filtered_data['date'] <= pd.to_datetime(end_date)]
            
            # Create new analytics instance with filtered data
            analytics = SalesAnalytics(filtered_data)
        
        metrics = analytics.calculate_key_metrics()
        response = MetricsResponse(
            **metrics,
            timestamp=datetime.now()
        )
        
        # Cache for 5 minutes
        redis_client.setex(cache_key, 300, response.json())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

@app.get("/api/v1/customer-segments")
async def get_customer_segments(
    n_segments: int = Query(4, ge=2, le=10, description="Number of customer segments")
):
    """
    Get customer segmentation analysis using RFM methodology
    """
    try:
        analytics = get_analytics_engine()
        rfm_data, segment_stats = analytics.customer_segmentation(n_segments)
        
        # Get CLV data
        clv_data = analytics.calculate_customer_lifetime_value()
        
        # Merge RFM and CLV data
        combined_data = rfm_data.merge(
            clv_data[['predicted_clv']], 
            left_index=True, 
            right_index=True
        )
        
        # Convert to response format
        segments = []
        for customer_id, row in combined_data.iterrows():
            segments.append(CustomerSegment(
                customer_id=customer_id,
                recency=int(row['recency']),
                frequency=int(row['frequency']),
                monetary=float(row['monetary']),
                segment_name=row['segment_name'],
                clv_prediction=float(row['predicted_clv'])
            ))
        
        # Get segment distribution
        segment_distribution = rfm_data['segment_name'].value_counts().to_dict()
        
        return {
            "segments": segments[:100],  # Return top 100 for API response
            "segment_distribution": segment_distribution,
            "segment_statistics": segment_stats.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in segmentation: {str(e)}")

@app.get("/api/v1/anomalies")
async def detect_sales_anomalies(
    lookback_days: int = Query(90, description="Number of days to analyze"),
    sensitivity: float = Query(0.05, ge=0.01, le=0.2, description="Anomaly detection sensitivity")
):
    """
    Detect anomalous sales patterns in recent data
    """
    try:
        analytics = get_analytics_engine()
        
        # Filter recent data
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent_data = analytics.data[analytics.data['date'] >= cutoff_date]
        
        # Run anomaly detection
        recent_analytics = SalesAnalytics(recent_data)
        anomalies = recent_analytics.detect_anomalies(contamination=sensitivity)
        
        # Format response
        anomaly_list = []
        for date_val, row in anomalies.iterrows():
            anomaly_list.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "total_revenue": float(row['total_amount_sum']),
                "transaction_count": int(row['total_amount_count']),
                "anomaly_score": float(row['anomaly_score']),
                "description": "Unusual sales pattern detected"
            })
        
        return {
            "anomalies": anomaly_list,
            "total_detected": len(anomaly_list),
            "analysis_period": f"{lookback_days} days",
            "sensitivity": sensitivity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@app.get("/api/v1/insights")
async def get_actionable_insights():
    """
    Get AI-generated actionable business insights
    """
    try:
        analytics = get_analytics_engine()
        raw_insights = analytics.generate_actionable_insights()
        
        # Convert to structured format
        insights = []
        for insight in raw_insights:
            # Determine priority based on keywords
            priority = "medium"
            action_required = False
            
            if "⚠️" in insight:
                priority = "high"
                action_required = True
            elif "🚀" in insight:
                priority = "low"
            elif "💡" in insight or "📊" in insight:
                priority = "medium"
                action_required = True
            
            # Determine insight type
            insight_type = "general"
            if "revenue" in insight.lower():
                insight_type = "revenue"
            elif "customer" in insight.lower():
                insight_type = "customer"
            elif "product" in insight.lower() or "inventory" in insight.lower():
                insight_type = "product"
            elif "staff" in insight.lower() or "day" in insight.lower():
                insight_type = "operational"
            
            insights.append(InsightResponse(
                insight_type=insight_type,
                message=insight,
                priority=priority,
                action_required=action_required
            ))
        
        return {
            "insights": insights,
            "generated_at": datetime.now(),
            "total_insights": len(insights)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@app.get("/api/v1/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """
    Get comprehensive dashboard data in a single call
    """
    try:
        analytics = get_analytics_engine()
        
        # Gather all components asynchronously
        metrics_task = asyncio.create_task(get_business_metrics())
        
        # Get product performance
        product_perf = analytics.product_performance_analysis(top_n=5)
        top_products = []
        for product_id, row in product_perf['top_revenue_products'].iterrows():
            top_products.append({
                "product_id": product_id,
                "revenue": float(row['total_amount']),
                "units_sold": int(row['quantity']),
                "velocity": float(row['daily_velocity'])
            })
        
        # Get customer segments
        rfm_data, _ = analytics.customer_segmentation()
        segment_dist = rfm_data['segment_name'].value_counts().to_dict()
        
        # Get recent anomalies
        anomalies = analytics.detect_anomalies()
        recent_anomalies = []
        for date_val, row in anomalies.tail(5).iterrows():
            recent_anomalies.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "revenue": float(row['total_amount_sum']),
                "severity": "high" if row['anomaly_score'] < -0.5 else "medium"
            })
        
        # Get insights
        insights_response = await get_actionable_insights()
        
        # Wait for metrics
        metrics = await metrics_task
        
        return DashboardData(
            metrics=metrics,
            top_products=top_products,
            customer_segments=segment_dist,
            recent_anomalies=recent_anomalies,
            insights=insights_response['insights']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dashboard: {str(e)}")

@app.post("/api/v1/transactions")
async def add_transaction(transaction: SalesTransaction):
    """
    Add a new sales transaction (for demo purposes)
    """
    try:
        # In production, this would save to database
        # For demo, just return success
        return {
            "status": "success",
            "message": "Transaction recorded",
            "transaction_id": f"TXN-{datetime.now().timestamp()}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding transaction: {str(e)}")

@app.get("/api/v1/forecast/{product_id}")
async def get_product_forecast(
    product_id: str,
    days_ahead: int = Query(30, ge=7, le=90, description="Number of days to forecast")
):
    """
    Get sales forecast for a specific product
    """
    try:
        # Simplified forecast logic for demo
        # In production, would use Prophet or similar
        analytics = get_analytics_engine()
        
        # Get historical data for product
        product_data = analytics.data[analytics.data['product_id'] == product_id]
        
        if len(product_data) == 0:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        # Calculate simple forecast based on trend
        daily_avg = product_data.groupby('date')['quantity'].sum().mean()
        weekly_pattern = product_data.groupby('day_of_week')['quantity'].mean()
        
        forecast = {
            "product_id": product_id,
            "forecast_period": f"{days_ahead} days",
            "predicted_units": int(daily_avg * days_ahead),
            "confidence_interval": {
                "lower": int(daily_avg * days_ahead * 0.8),
                "upper": int(daily_avg * days_ahead * 1.2)
            },
            "weekly_pattern": weekly_pattern.to_dict(),
            "recommendation": "Maintain current inventory levels" if daily_avg > 5 else "Consider promotion"
        }
        
        return forecast
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

# Websocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/live-metrics")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics updates
    """
    await websocket.accept()
    try:
        while True:
            # Send updated metrics every 5 seconds
            metrics = await get_business_metrics()
            await websocket.send_json(metrics.dict())
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)