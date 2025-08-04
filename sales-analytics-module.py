"""
Small Business Intelligence Hub - Sales Analytics Module
Author: Justin Christopher Weaver
Description: Core analytics engine for sales data processing and insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class SalesAnalytics:
    """
    Comprehensive sales analytics engine for small businesses
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analytics engine with sales data
        
        Args:
            data: DataFrame with columns [date, product_id, customer_id, 
                  quantity, unit_price, total_amount, category]
        """
        self.data = data
        self.data['date'] = pd.to_datetime(self.data['date'])
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess and enrich data with calculated fields"""
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['week_of_year'] = self.data['date'].dt.isocalendar().week
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6])
    
    def calculate_key_metrics(self) -> Dict:
        """
        Calculate essential business metrics
        
        Returns:
            Dictionary containing key performance indicators
        """
        metrics = {
            'total_revenue': self.data['total_amount'].sum(),
            'total_transactions': len(self.data),
            'average_order_value': self.data['total_amount'].mean(),
            'unique_customers': self.data['customer_id'].nunique(),
            'unique_products': self.data['product_id'].nunique(),
            'revenue_per_customer': self.data['total_amount'].sum() / self.data['customer_id'].nunique()
        }
        
        # Calculate growth metrics
        if len(self.data) > 30:
            last_30_days = self.data[self.data['date'] >= (datetime.now() - timedelta(days=30))]
            prev_30_days = self.data[
                (self.data['date'] >= (datetime.now() - timedelta(days=60))) &
                (self.data['date'] < (datetime.now() - timedelta(days=30)))
            ]
            
            if len(prev_30_days) > 0:
                metrics['revenue_growth_30d'] = (
                    (last_30_days['total_amount'].sum() - prev_30_days['total_amount'].sum()) / 
                    prev_30_days['total_amount'].sum() * 100
                )
        
        return metrics
    
    def customer_segmentation(self, n_segments: int = 4) -> pd.DataFrame:
        """
        Perform RFM (Recency, Frequency, Monetary) customer segmentation
        
        Args:
            n_segments: Number of customer segments to create
            
        Returns:
            DataFrame with customer segments and characteristics
        """
        current_date = self.data['date'].max()
        
        # Calculate RFM metrics
        rfm = self.data.groupby('customer_id').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'customer_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).rename(columns={
            'date': 'recency',
            'customer_id': 'frequency',
            'total_amount': 'monetary'
        })
        
        # Normalize features
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_segments, random_state=42)
        rfm['segment'] = kmeans.fit_predict(rfm_scaled)
        
        # Define segment names based on characteristics
        segment_names = {
            0: 'Champions',
            1: 'Loyal Customers',
            2: 'At Risk',
            3: 'New Customers'
        }
        
        rfm['segment_name'] = rfm['segment'].map(segment_names)
        
        # Calculate segment statistics
        segment_stats = rfm.groupby('segment_name').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        return rfm, segment_stats
    
    def detect_anomalies(self, contamination: float = 0.05) -> pd.DataFrame:
        """
        Detect anomalous sales patterns using Isolation Forest
        
        Args:
            contamination: Expected proportion of anomalies in dataset
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Prepare features for anomaly detection
        daily_stats = self.data.groupby('date').agg({
            'total_amount': ['sum', 'mean', 'std', 'count'],
            'quantity': ['sum', 'mean'],
            'customer_id': 'nunique'
        })
        
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        daily_stats = daily_stats.fillna(0)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        daily_stats['anomaly'] = iso_forest.fit_predict(daily_stats)
        daily_stats['anomaly_score'] = iso_forest.score_samples(daily_stats)
        
        # Flag anomalies
        daily_stats['is_anomaly'] = daily_stats['anomaly'] == -1
        
        return daily_stats[daily_stats['is_anomaly']]
    
    def product_performance_analysis(self, top_n: int = 10) -> Dict:
        """
        Analyze product performance metrics
        
        Args:
            top_n: Number of top products to return
            
        Returns:
            Dictionary with product performance insights
        """
        product_stats = self.data.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'customer_id': 'nunique',
            'date': 'count'
        }).rename(columns={'date': 'transaction_count'})
        
        product_stats['avg_price'] = product_stats['total_amount'] / product_stats['quantity']
        product_stats['revenue_per_customer'] = product_stats['total_amount'] / product_stats['customer_id']
        
        # Calculate product velocity (sales rate)
        days_active = (self.data['date'].max() - self.data['date'].min()).days + 1
        product_stats['daily_velocity'] = product_stats['quantity'] / days_active
        
        # Identify top performers
        top_revenue = product_stats.nlargest(top_n, 'total_amount')
        top_quantity = product_stats.nlargest(top_n, 'quantity')
        top_velocity = product_stats.nlargest(top_n, 'daily_velocity')
        
        return {
            'top_revenue_products': top_revenue,
            'top_selling_products': top_quantity,
            'fastest_moving_products': top_velocity,
            'product_summary': product_stats.describe()
        }
    
    def time_series_decomposition(self) -> Dict:
        """
        Decompose sales time series to identify trends and patterns
        
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        # Aggregate daily sales
        daily_sales = self.data.groupby('date')['total_amount'].sum().reset_index()
        daily_sales = daily_sales.set_index('date')
        
        # Calculate moving averages
        daily_sales['ma_7'] = daily_sales['total_amount'].rolling(window=7).mean()
        daily_sales['ma_30'] = daily_sales['total_amount'].rolling(window=30).mean()
        
        # Calculate day-of-week seasonality
        dow_pattern = self.data.groupby('day_of_week')['total_amount'].mean()
        
        # Calculate monthly seasonality
        monthly_pattern = self.data.groupby('month')['total_amount'].mean()
        
        return {
            'daily_trend': daily_sales,
            'day_of_week_pattern': dow_pattern,
            'monthly_pattern': monthly_pattern,
            'peak_day': dow_pattern.idxmax(),
            'peak_month': monthly_pattern.idxmax()
        }
    
    def calculate_customer_lifetime_value(self, time_period_days: int = 365) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV) predictions
        
        Args:
            time_period_days: Time period for CLV calculation
            
        Returns:
            DataFrame with CLV predictions per customer
        """
        # Calculate customer metrics
        customer_metrics = self.data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        })
        
        customer_metrics.columns = ['total_revenue', 'avg_order_value', 
                                   'order_count', 'first_purchase', 'last_purchase']
        
        # Calculate customer age and purchase frequency
        customer_metrics['customer_age_days'] = (
            customer_metrics['last_purchase'] - customer_metrics['first_purchase']
        ).dt.days + 1
        
        customer_metrics['purchase_frequency'] = (
            customer_metrics['order_count'] / customer_metrics['customer_age_days']
        )
        
        # Simple CLV calculation
        customer_metrics['predicted_clv'] = (
            customer_metrics['avg_order_value'] * 
            customer_metrics['purchase_frequency'] * 
            time_period_days
        )
        
        # Segment by CLV
        customer_metrics['clv_segment'] = pd.qcut(
            customer_metrics['predicted_clv'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'VIP']
        )
        
        return customer_metrics
    
    def generate_actionable_insights(self) -> List[str]:
        """
        Generate actionable business insights based on analytics
        
        Returns:
            List of actionable recommendations
        """
        insights = []
        
        # Analyze key metrics
        metrics = self.calculate_key_metrics()
        
        # Revenue insights
        if 'revenue_growth_30d' in metrics:
            if metrics['revenue_growth_30d'] < 0:
                insights.append(
                    f"⚠️ Revenue declined by {abs(metrics['revenue_growth_30d']):.1f}% "
                    "in the last 30 days. Consider promotional campaigns."
                )
            elif metrics['revenue_growth_30d'] > 20:
                insights.append(
                    f"🚀 Excellent growth! Revenue increased by {metrics['revenue_growth_30d']:.1f}% "
                    "in the last 30 days."
                )
        
        # Customer insights
        rfm, segment_stats = self.customer_segmentation()
        at_risk_count = len(rfm[rfm['segment_name'] == 'At Risk'])
        if at_risk_count > len(rfm) * 0.2:
            insights.append(
                f"⚠️ {at_risk_count} customers ({at_risk_count/len(rfm)*100:.1f}%) are at risk. "
                "Launch a win-back campaign."
            )
        
        # Product insights
        product_perf = self.product_performance_analysis()
        top_products = product_perf['top_revenue_products'].head(3).index.tolist()
        insights.append(
            f"💡 Focus inventory on top performers: {', '.join(map(str, top_products))}"
        )
        
        # Time-based insights
        time_patterns = self.time_series_decomposition()
        peak_day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday'][time_patterns['peak_day']]
        insights.append(
            f"📊 {peak_day_name} is your busiest day. Ensure adequate staffing."
        )
        
        # Anomaly insights
        anomalies = self.detect_anomalies()
        if len(anomalies) > 0:
            insights.append(
                f"🔍 Detected {len(anomalies)} unusual sales patterns. Review for opportunities."
            )
        
        return insights


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create synthetic sales data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n_transactions = 10000
    
    sample_data = pd.DataFrame({
        'date': np.random.choice(dates, n_transactions),
        'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_transactions),
        'customer_id': np.random.choice([f'C{i:03d}' for i in range(100)], n_transactions),
        'quantity': np.random.randint(1, 10, n_transactions),
        'unit_price': np.random.uniform(10, 100, n_transactions),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_transactions)
    })
    
    sample_data['total_amount'] = sample_data['quantity'] * sample_data['unit_price']
    
    # Initialize analytics engine
    analytics = SalesAnalytics(sample_data)
    
    # Run analytics
    print("=== KEY METRICS ===")
    metrics = analytics.calculate_key_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:,.2f}")
    
    print("\n=== CUSTOMER SEGMENTS ===")
    rfm, segments = analytics.customer_segmentation()
    print(segments)
    
    print("\n=== ACTIONABLE INSIGHTS ===")
    insights = analytics.generate_actionable_insights()
    for insight in insights:
        print(insight)
