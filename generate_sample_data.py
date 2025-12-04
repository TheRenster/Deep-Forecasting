"""
Generate sample time series data for testing the forecasting app
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(n_periods=120, start_date='2020-01-01'):
    """
    Generate synthetic sales data with trend and seasonality
    
    Parameters:
    - n_periods: Number of time periods (months)
    - start_date: Start date for the series
    """
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_periods, freq='MS')
    
    # Base trend (increasing over time)
    trend = np.linspace(1000, 2000, n_periods)
    
    # Seasonal component (12-month seasonality)
    seasonal = 300 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
    
    # Random noise
    noise = np.random.normal(0, 50, n_periods)
    
    # Combine components
    sales = trend + seasonal + noise
    
    # Ensure no negative values
    sales = np.maximum(sales, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2)
    })
    
    # Add some missing values (5%)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'Sales'] = np.nan
    
    return df

def generate_temperature_data(n_periods=365*3, start_date='2021-01-01'):
    """
    Generate synthetic daily temperature data
    
    Parameters:
    - n_periods: Number of days
    - start_date: Start date for the series
    """
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
    
    # Base annual cycle (365-day seasonality)
    base_temp = 60  # Average temperature
    seasonal_amplitude = 30  # Temperature variation
    seasonal = seasonal_amplitude * np.sin((np.arange(n_periods) * 2 * np.pi / 365) - np.pi/2)
    
    # Weekly pattern (slightly warmer on weekends)
    weekly_pattern = 2 * np.sin(np.arange(n_periods) * 2 * np.pi / 7)
    
    # Random daily variation
    noise = np.random.normal(0, 5, n_periods)
    
    # Slight warming trend
    trend = np.linspace(0, 3, n_periods)
    
    # Combine
    temperature = base_temp + seasonal + weekly_pattern + noise + trend
    
    df = pd.DataFrame({
        'Date': dates,
        'Temperature_F': temperature.round(1)
    })
    
    return df

def generate_website_traffic(n_periods=104, start_date='2022-01-01'):
    """
    Generate synthetic weekly website traffic data
    
    Parameters:
    - n_periods: Number of weeks
    - start_date: Start date for the series
    """
    # Generate date range (weekly)
    dates = pd.date_range(start=start_date, periods=n_periods, freq='W')
    
    # Exponential growth trend
    trend = 5000 * np.exp(np.linspace(0, 1, n_periods))
    
    # Annual seasonality (52-week cycle)
    seasonal = 1000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52)
    
    # Random variation
    noise = np.random.normal(0, 500, n_periods)
    
    # Occasional spikes (marketing campaigns)
    spikes = np.zeros(n_periods)
    spike_indices = np.random.choice(n_periods, size=5, replace=False)
    spikes[spike_indices] = np.random.uniform(1000, 3000, 5)
    
    # Combine
    traffic = trend + seasonal + noise + spikes
    traffic = np.maximum(traffic, 0).round(0)
    
    df = pd.DataFrame({
        'Week_Start': dates,
        'Visitors': traffic
    })
    
    return df

def generate_stock_price(n_periods=252*2, start_date='2022-01-01'):
    """
    Generate synthetic stock price data (random walk with drift)
    
    Parameters:
    - n_periods: Number of trading days
    - start_date: Start date for the series
    """
    # Generate date range (business days)
    dates = pd.date_range(start=start_date, periods=n_periods, freq='B')
    
    # Random walk with drift
    initial_price = 100
    drift = 0.0005  # Slight upward drift
    volatility = 0.02
    
    returns = np.random.normal(drift, volatility, n_periods)
    price_levels = initial_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Close_Price': price_levels.round(2),
        'Volume': np.random.uniform(1e6, 5e6, n_periods).round(0)
    })
    
    return df

# Generate all sample datasets
if __name__ == "__main__":
    print("Generating sample datasets...")
    
    # 1. Monthly sales data
    sales_df = generate_sales_data(n_periods=120)
    sales_df.to_csv('sample_sales_data.csv', index=False)
    print(f"✓ Generated sample_sales_data.csv ({len(sales_df)} rows)")
    
    # 2. Daily temperature data
    temp_df = generate_temperature_data(n_periods=365*3)
    temp_df.to_csv('sample_temperature_data.csv', index=False)
    print(f"✓ Generated sample_temperature_data.csv ({len(temp_df)} rows)")
    
    # 3. Weekly website traffic
    traffic_df = generate_website_traffic(n_periods=104)
    traffic_df.to_csv('sample_traffic_data.csv', index=False)
    print(f"✓ Generated sample_traffic_data.csv ({len(traffic_df)} rows)")
    
    # 4. Daily stock prices
    stock_df = generate_stock_price(n_periods=252*2)
    stock_df.to_csv('sample_stock_data.csv', index=False)
    print(f"✓ Generated sample_stock_data.csv ({len(stock_df)} rows)")
    
    print("\n✅ All sample datasets generated successfully!")
    print("\nDataset summaries:")
    print("\n1. Sales Data (Monthly):")
    print(sales_df.describe())
    print("\n2. Temperature Data (Daily):")
    print(temp_df.describe())
    print("\n3. Traffic Data (Weekly):")
    print(traffic_df.describe())
    print("\n4. Stock Data (Daily):")
    print(stock_df.describe())
