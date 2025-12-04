import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import sktime for ETS
try:
    from sktime.forecasting.ets import AutoETS
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Forecasting Friend",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" # Changed to collapsed for a minimal start
)

# Enhanced CSS with modern styling and new colors
st.markdown("""
    <style>
    /* Main styling - NEW BACKGROUND COLOR #0e1117 */
    .main {
        background: #0e1117; 
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: white; /* Changed subtitle text color for better contrast on dark background */
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #f8a6da 0%, #f6b173 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Metric cards - NEW BOX COLOR #ffd7ef */
    div[data-testid="stMetric"] {
        background: #ffd7ef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar - Kept gradient, as contents are white */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8a6da 0%, #f6b173  100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* File uploader - NEW BOX COLOR #ffd7ef */
    div[data-testid="stFileUploader"] {
        background: #ffd7ef;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Dataframe - NEW BOX COLOR #ffd7ef */
    div[data-testid="stDataFrame"] {
        background: #ffd7ef;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success/Info boxes - NEW BOX COLOR #ffd7ef */
    .stSuccess, .stInfo {
        background-color: #ffd7ef !important; 
        color: black !important; /* Ensure text is readable */
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .stSuccess>div, .stInfo>div {
        color: black !important;
    }
    .stSuccess>div>svg, .stInfo>div>svg {
        fill: #667eea !important;
    }
    
    /* Expander - Adjusted for dark background */
    .streamlit-expanderHeader {
        background: #333; /* Darker header for contrast */
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Text color for the main content area */
    h1, h2, h3, h4, p, label, .stMarkdown, div[data-testid="stMetricLabel"] {
        color: white;
    }
    
    </style>
""", unsafe_allow_html=True)

# Helper Functions (Keep as is)
def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_lag_features(df, target_col, n_lags=12):
    """Create lag features for ML models"""
    data = df.copy()
    for i in range(1, n_lags + 1):
        data[f'lag_{i}'] = data[target_col].shift(i)
    
    # Add rolling statistics
    data['rolling_mean_3'] = data[target_col].shift(1).rolling(window=3).mean()
    data['rolling_mean_6'] = data[target_col].shift(1).rolling(window=6).mean()
    data['rolling_std_3'] = data[target_col].shift(1).rolling(window=3).std()
    
    return data.dropna()

def train_holt_winters(train_data, forecast_horizon, seasonal_periods=12):
    """Train Holt-Winters Exponential Smoothing model"""
    try:
        model = ExponentialSmoothing(
            train_data,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_horizon)
        return forecast, "Success"
    except Exception as e:
        return None, str(e)

def train_auto_ets(train_data, forecast_horizon):
    """Train AutoETS model using sktime"""
    try:
        # Convert to period index for sktime
        train_period = train_data.copy()
        train_period.index = pd.PeriodIndex(train_period.index, freq='M')
        
        model = AutoETS(auto=True, sp=12, n_jobs=-1)
        model.fit(train_period)
        
        # Create forecast horizon
        forecast = model.predict(fh=np.arange(1, forecast_horizon + 1))
        
        # Convert back to datetime index
        forecast.index = pd.date_range(start=train_data.index[-1] + pd.DateOffset(months=1), 
                                       periods=forecast_horizon, freq='MS')
        return forecast.values, "Success"
    except Exception as e:
        return None, str(e)

def train_sarima(train_data, forecast_horizon, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Train SARIMA model"""
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order
        )
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.forecast(steps=forecast_horizon)
        return forecast, "Success"
    except Exception as e:
        return None, str(e)

def train_random_forest(train_df, test_df, target_col, n_lags=12):
    """Train Random Forest model"""
    try:
        train_features = create_lag_features(train_df, target_col, n_lags)
        test_features = create_lag_features(test_df, target_col, n_lags)
        
        feature_cols = [col for col in train_features.columns if col != target_col]
        
        X_train = train_features[feature_cols]
        y_train = train_features[target_col]
        X_test = test_features[feature_cols]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        
        return forecast, "Success"
    except Exception as e:
        return None, str(e)

def train_gradient_boosting(train_df, test_df, target_col, n_lags=12):
    """Train Gradient Boosting model"""
    try:
        train_features = create_lag_features(train_df, target_col, n_lags)
        test_features = create_lag_features(test_df, target_col, n_lags)
        
        feature_cols = [col for col in train_features.columns if col != target_col]
        
        X_train = train_features[feature_cols]
        y_train = train_features[target_col]
        X_test = test_features[feature_cols]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        
        return forecast, "Success"
    except Exception as e:
        return None, str(e)

# Main App
# Main App
def main():
    # 1. Define the filename for the Forecasting App
    IMAGE_FILENAME = 'forecasting_friend.png'

    # 2. Construct the path relative to the current working directory
    # This works both locally and on Streamlit Cloud
    image_path = Path.cwd() / IMAGE_FILENAME

    # App title with your custom logo
    if image_path.is_file():
        st.image(str(image_path), use_column_width=True)  # Fill container width
        st.markdown(
            """
            <div style="text-align:center; margin-top: -10px;">
                <p style="color: #e5e5e5; font-size: 18px;">
                    Advanced Forecasting with Multiple Models & Real-Time Comparison
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Fallback if image not found
        st.error(f"Image not found at path: {image_path}")
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="font-family: 'Inter', sans-serif; font-size: 44px; font-weight: 700; color: #fff; margin: 0;">
                    Time Series Forecasting App
                </h1>
                <p style="color: #ccc; font-size: 18px; margin-top: 10px;">
                    Advanced Forecasting with Multiple Models & Real-Time Comparison
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Sidebar - Simulating the grid icon's function with a collapsed sidebar
    with st.sidebar:
        # The sidebar starts collapsed, and the expander gives the "details" of the app
        st.markdown("## App Information")
        st.markdown("---")
        
        # About section (expanded by default to act as the primary configuration visible when opened)
        with st.expander("About This App (Click for Info)", expanded=True):
            st.markdown("""
            ### What This App Does
            Upload time series data and compare multiple forecasting approaches:
            
            **Econometric Models:**
            - Holt-Winters (Exponential Smoothing)
            - SARIMA (Seasonal ARIMA)
            - AutoETS (Automatic Error-Trend-Seasonal)
            
            **Machine Learning Models:**
            - Random Forest (Ensemble Learning)
            - Gradient Boosting (Sequential Learning)
            
            **Evaluation Metrics:**
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - MAPE: Mean Absolute Percentage Error
            
            **Created by:** [Your Name]
            """)
        
        st.markdown("---")
        st.info("The configuration steps are primarily in the main page (Steps 1-5). Use this sidebar for App Info.")


    # File Upload Section
    st.markdown("### Step 1: Upload Your Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())
            col4.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show data preview
            with st.expander("Data Preview & Info", expanded=True):
                tab1, tab2 = st.tabs(["Data Preview", "Quick Stats"])
                
                with tab1:
                    st.dataframe(df.head(10), )
                
                with tab2:
                    st.dataframe(df.describe(), use_column_width=True)
            
            # Column Selection
            st.markdown("### Step 2: Configure Your Forecast")
            
            col1, col2 = st.columns(2)
            
            with col1:
                date_col = st.selectbox("Select Date/Time Column", df.columns, key="date_col")
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                target_col = st.selectbox("Select Target Variable", numeric_cols, key="target_col")
            
            if date_col and target_col:
                # Parse dates
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(date_col)
                    df = df.set_index(date_col)
                    st.success("Date column parsed and data sorted successfully!")
                except Exception as e:
                    st.error(f"Error parsing date column: {e}")
                    return
                
                # Handle missing values
                st.markdown("### Step 3: Data Preprocessing")
                missing_count = df[target_col].isnull().sum()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if missing_count > 0:
                        st.warning(f"Found {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
                        handling_method = st.radio(
                            "Choose handling method:",
                            ["Drop rows with missing values", "Forward fill", "Backward fill", "Linear interpolation"],
                            horizontal=True
                        )
                        
                        if handling_method == "Drop rows with missing values":
                            df = df.dropna(subset=[target_col])
                        elif handling_method == "Forward fill":
                            df[target_col] = df[target_col].ffill()
                        elif handling_method == "Backward fill":
                            df[target_col] = df[target_col].bfill()
                        else:
                            df[target_col] = df[target_col].interpolate(method='linear')
                        
                        st.success(f"Applied: {handling_method}")
                    else:
                        st.success("No missing values detected!")
                
                with col2:
                    st.metric("Clean Records", len(df))
                
                # Date range filter
                use_filter = st.checkbox("Filter data by date range or last N periods")
                
                if use_filter:
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        filter_option = st.radio("Filter by:", ["Date range", "Last N periods"])
                    
                    with filter_col2:
                        if filter_option == "Date range":
                            min_date = df.index.min()
                            max_date = df.index.max()
                            start_date = st.date_input("Start date", min_date)
                            end_date = st.date_input("End date", max_date)
                            df = df[start_date:end_date]
                        else:
                            n_periods = st.number_input("Number of recent periods", min_value=10, max_value=len(df), value=min(100, len(df)))
                            df = df.tail(n_periods)
                    
                    st.info(f"Filtered to {len(df)} records")
                
                # Train/Test Configuration
                st.markdown("### Step 4: Model Configuration")
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    test_size = st.slider("Test set size (%)", 10, 40, 20)
                    test_periods = int(len(df) * test_size / 100)
                
                with config_col2:
                    forecast_horizon = st.number_input(
                        "Forecast horizon",
                        min_value=1,
                        max_value=test_periods,
                        value=min(12, test_periods)
                    )
                
                with config_col3:
                    seasonal_periods = st.number_input("Seasonal periods", min_value=2, max_value=24, value=12)
                
                train_data = df[target_col].iloc[:-test_periods]
                test_data = df[target_col].iloc[-test_periods:]
                
                # Show split info
                split_col1, split_col2, split_col3 = st.columns(3)
                split_col1.metric("Train Size", len(train_data))
                split_col2.metric("Test Size", len(test_data))
                split_col3.metric("Forecast Steps", forecast_horizon)
                
                # Model Selection
                st.markdown("### Step 5: Select Models to Train")
                
                model_col1, model_col2 = st.columns(2)
                
                with model_col1:
                    st.markdown("**Econometric Models**")
                    use_hw = st.checkbox("Holt-Winters Exponential Smoothing", value=True)
                    use_sarima = st.checkbox("SARIMA (Seasonal ARIMA)", value=True)
                    if SKTIME_AVAILABLE:
                        use_auto_ets = st.checkbox("AutoETS (Automatic Selection)", value=True)
                    else:
                        use_auto_ets = False
                        st.info("Install sktime for AutoETS: `pip install sktime`")
                
                with model_col2:
                    st.markdown("**Machine Learning Models**")
                    use_rf = st.checkbox("Random Forest Regressor", value=True)
                    use_gb = st.checkbox("Gradient Boosting Regressor", value=True)
                    
                    if use_rf or use_gb:
                        n_lags = st.number_input("Number of lags for ML", min_value=3, max_value=24, value=12)
                
                # Train Models Button
                st.markdown("---")
                train_button = st.button("TRAIN MODELS & GENERATE FORECASTS", use_column_width=True)
                
                if train_button:
                    results = {}
                    
                    with st.spinner("Training models... Please wait..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_models = sum([use_hw, use_sarima, use_auto_ets, use_rf, use_gb])
                        current_model = 0
                        
                        # Holt-Winters
                        if use_hw:
                            status_text.text("Training Holt-Winters Exponential Smoothing...")
                            forecast, status = train_holt_winters(train_data, forecast_horizon, seasonal_periods)
                            if status == "Success":
                                results['Holt-Winters'] = forecast
                            else:
                                st.warning(f"Holt-Winters failed: {status}")
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                        
                        # SARIMA
                        if use_sarima:
                            status_text.text("Training SARIMA...")
                            forecast, status = train_sarima(train_data, forecast_horizon, seasonal_order=(1,1,1,seasonal_periods))
                            if status == "Success":
                                results['SARIMA'] = forecast
                            else:
                                st.warning(f"SARIMA failed: {status}")
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                        
                        # AutoETS
                        if use_auto_ets and SKTIME_AVAILABLE:
                            status_text.text("Training AutoETS...")
                            forecast, status = train_auto_ets(train_data, forecast_horizon)
                            if status == "Success":
                                results['AutoETS'] = forecast
                            else:
                                st.warning(f"AutoETS failed: {status}")
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                        
                        # Random Forest
                        if use_rf:
                            status_text.text("Training Random Forest...")
                            train_df = df.iloc[:-test_periods].copy()
                            test_df = df.iloc[-test_periods:].copy()
                            forecast, status = train_random_forest(train_df, test_df, target_col, n_lags)
                            if status == "Success":
                                # Random Forest, GB predictions are array-like, need to ensure proper length and indexing
                                # Since train_random_forest/train_gradient_boosting is designed to return predictions 
                                # for the length of X_test, slicing to forecast_horizon is correct for ML models
                                results['Random Forest'] = forecast[:forecast_horizon] 
                            else:
                                st.warning(f"Random Forest failed: {status}")
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                        
                        # Gradient Boosting
                        if use_gb:
                            status_text.text("Training Gradient Boosting...")
                            train_df = df.iloc[:-test_periods].copy()
                            test_df = df.iloc[-test_periods:].copy()
                            forecast, status = train_gradient_boosting(train_df, test_df, target_col, n_lags)
                            if status == "Success":
                                results['Gradient Boosting'] = forecast[:forecast_horizon]
                            else:
                                st.warning(f"Gradient Boosting failed: {status}")
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    if results:
                        st.success(f"Successfully trained {len(results)} models!")
                        
                        # Evaluation
                        st.markdown("## Model Performance Evaluation")
                        
                        # Ensure actual_values is properly indexed for time series models
                        actual_values = test_data.iloc[:forecast_horizon].values
                        metrics_data = []
                        
                        for model_name, forecast in results.items():
                            rmse = np.sqrt(mean_squared_error(actual_values, forecast))
                            mae = mean_absolute_error(actual_values, forecast)
                            mape = calculate_mape(actual_values, forecast)
                            
                            metrics_data.append({
                                'Model': model_name,
                                'RMSE': round(rmse, 4),
                                'MAE': round(mae, 4),
                                'MAPE (%)': round(mape, 2)
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df = metrics_df.sort_values('RMSE')
                        
                        # Display metrics with highlighting
                        st.dataframe(
                            metrics_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen'),
                            use_column_width=True
                        )
                        
                        # Best model
                        best_model = metrics_df.iloc[0]['Model']
                        best_rmse = metrics_df.iloc[0]['RMSE']
                        best_mape = metrics_df.iloc[0]['MAPE (%)']
                        
                        st.success(f" **Best Model:** {best_model} (RMSE: {best_rmse:.4f}, MAPE: {best_mape:.2f}%)")
                        
                        # Visualizations
                        st.markdown("## Forecast Visualizations")
                        
                        # Create tabs for different views
                        viz_tabs = st.tabs(["Individual Models", "Model Comparison", "Metrics Charts"])
                        
                        # Determine the index for the forecast results
                        forecast_index = test_data.index[:forecast_horizon]
                        
                        with viz_tabs[0]:
                            # Individual model plots
                            for model_name, forecast in results.items():
                                fig, ax = plt.subplots(figsize=(14, 6))
                                
                                ax.plot(train_data.index, train_data.values, label='Training Data', 
                                       color='#667eea', alpha=0.7, linewidth=2)
                                ax.plot(forecast_index, actual_values, 
                                       label='Actual', color='#2ecc71', marker='o', linewidth=2.5, markersize=6)
                                ax.plot(forecast_index, forecast, 
                                       label=f'{model_name} Forecast', color='#e74c3c', marker='x', 
                                       linewidth=2.5, linestyle='--', markersize=8)
                                
                                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                                ax.set_ylabel(target_col, fontsize=12, fontweight='bold')
                                ax.set_title(f'{model_name} - Forecast Performance', fontsize=14, fontweight='bold', pad=20)
                                ax.legend(fontsize=11, loc='best')
                                ax.grid(True, alpha=0.3, linestyle='--')
                                
                                # Set background and text color for plots
                                fig.patch.set_facecolor('#333')
                                ax.set_facecolor('#1e1e1e')
                                ax.xaxis.label.set_color('white')
                                ax.yaxis.label.set_color('white')
                                ax.title.set_color('white')
                                ax.tick_params(axis='x', colors='white', rotation=45)
                                ax.tick_params(axis='y', colors='white')
                                
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                plt.close()
                        
                        with viz_tabs[1]:
                            # Comparison plot
                            fig, ax = plt.subplots(figsize=(16, 8))
                            
                            ax.plot(forecast_index, actual_values, 
                                   label='Actual', color='white', marker='o', linewidth=3.5, markersize=8, zorder=10) # Changed actual line color to white
                            
                            colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
                            for idx, (model_name, forecast) in enumerate(results.items()):
                                ax.plot(forecast_index, forecast, 
                                       label=f'{model_name}', marker='x', linewidth=2.5, 
                                       linestyle='--', color=colors[idx % len(colors)], markersize=7)
                            
                            ax.set_xlabel('Date', fontsize=13, fontweight='bold')
                            ax.set_ylabel(target_col, fontsize=13, fontweight='bold')
                            ax.set_title('All Models - Performance Comparison', fontsize=16, fontweight='bold', pad=20)
                            ax.legend(fontsize=12, loc='best', framealpha=0.9)
                            ax.grid(True, alpha=0.3, linestyle='--')
                            
                            # Set background and text color for plots
                            fig.patch.set_facecolor('#333')
                            ax.set_facecolor('#1e1e1e')
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            ax.title.set_color('white')
                            ax.tick_params(axis='x', colors='white', rotation=45)
                            ax.tick_params(axis='y', colors='white')
                            
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            plt.close()
                        
                        with viz_tabs[2]:
                            # Metrics comparison
                            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                            
                            metrics_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[0], 
                                          color='#667eea', legend=False, width=0.7)
                            axes[0].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
                            axes[0].set_ylabel('RMSE', fontsize=11)
                            axes[0].tick_params(axis='x', rotation=45)
                            axes[0].grid(axis='y', alpha=0.3)
                            
                            metrics_df.plot(x='Model', y='MAE', kind='bar', ax=axes[1], 
                                          color='#e74c3c', legend=False, width=0.7)
                            axes[1].set_title('MAE Comparison', fontsize=13, fontweight='bold')
                            axes[1].set_ylabel('MAE', fontsize=11)
                            axes[1].tick_params(axis='x', rotation=45)
                            axes[1].grid(axis='y', alpha=0.3)
                            
                            metrics_df.plot(x='Model', y='MAPE (%)', kind='bar', ax=axes[2], 
                                          color='#2ecc71', legend=False, width=0.7)
                            axes[2].set_title('MAPE Comparison', fontsize=13, fontweight='bold')
                            axes[2].set_ylabel('MAPE (%)', fontsize=11)
                            axes[2].tick_params(axis='x', rotation=45)
                            axes[2].grid(axis='y', alpha=0.3)
                            
                            # Apply dark theme to metric charts
                            for ax in axes:
                                ax.set_facecolor('#1e1e1e')
                                ax.patch.set_facecolor('#1e1e1e')
                                ax.xaxis.label.set_color('white')
                                ax.yaxis.label.set_color('white')
                                ax.title.set_color('white')
                                ax.tick_params(axis='x', colors='white')
                                ax.tick_params(axis='y', colors='white')
                            fig.patch.set_facecolor('#333')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Download results
                        st.markdown("## Download Results")
                        
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            csv = metrics_df.to_csv(index=False)
                            st.download_button(
                                label="Download Metrics (CSV)",
                                data=csv,
                                file_name="forecast_metrics.csv",
                                mime="text/csv"
                            )
                        
                        with download_col2:
                            forecast_df = pd.DataFrame(results, index=forecast_index)
                            forecast_csv = forecast_df.to_csv()
                            st.download_button(
                                label="Download Forecasts (CSV)",
                                data=forecast_csv,
                                file_name="forecasts.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("No models were successfully trained. Please check your data and settings.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV has a date column and at least one numeric column.")
    
    else:
        # Landing page
        st.markdown("""
        <div style='background: #f7aba7; padding: 3rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0;'>
            <h2 style='text-align: center; color: #0e1117;'> Welcome to Time Series Forecasting Pro</h2>
            <p style='text-align: center; font-size: 1.1rem; color: #0e1117;'>
                Upload your CSV file to get started with advanced time series forecasting
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights (adjusted for dark background)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: #f6b173 ; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #0e1117; text-align: center;'> Multiple Models</h3>
                <p style='text-align: center; color: black;'>Compare 5 different forecasting algorithms including econometric and ML approaches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #f6b173 ; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #0e1117; text-align: center;'> Visual Analytics</h3>
                <p style='text-align: center; color: black;'>Interactive charts and side-by-side model comparisons with performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: #f6b173 ; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #0e1117; text-align: center;'> Easy to Use</h3>
                <p style='text-align: center; color: black;'>Simple upload, configure, and train workflow with comprehensive documentation</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # How to use section (adjusted for dark background)
        with st.expander(" How to Use This App", expanded=False):
            st.markdown("""
            ### Quick Start Guide
            
            **Step 1: Upload Data**
            - Click the file uploader above
            - Select a CSV file with time series data
            - Your CSV should have a date column and at least one numeric column
            
            **Step 2: Configure**
            - Select your date/time column
            - Choose your target variable (what you want to forecast)
            - Handle any missing values
            - Optionally filter your date range
            
            **Step 3: Set Parameters**
            - Choose train/test split percentage
            - Set forecast horizon (how many periods ahead)
            - Select seasonal periods (12 for monthly, 7 for daily, etc.)
            
            **Step 4: Select Models**
            - Choose which models to train (at least 1 econometric + 1 ML)
            - Configure number of lags for ML models
            
            **Step 5: Train & Compare**
            - Click "Train Models" button
            - Wait for training to complete
            - Review metrics and visualizations
            - Download results if needed
            
            ### Sample Data Format
            
            ```csv
            Date,Sales
            2020-01-01,1200
            2020-02-01,1350
            2020-03-01,1280
            ...
            ```
            
            ### Understanding Metrics
            
            - **RMSE (Root Mean Squared Error)**: Measures average prediction error. Lower is better. Penalizes large errors more.
            - **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual. Lower is better. More robust to outliers.
            - **MAPE (Mean Absolute Percentage Error)**: Percentage-based error. Lower is better. Easy to interpret (e.g., 5% error).
            
            ### Model Selection Guide
            
            **Holt-Winters**: Best for data with clear trend and seasonal patterns. Fast and interpretable.
            
            **SARIMA**: Best for complex seasonal patterns. Handles multiple seasonal cycles well.
            
            **AutoETS**: Automatically selects best exponential smoothing model. Good for quick analysis.
            
            **Random Forest**: Best for non-linear relationships and irregular patterns. Very robust.
            
            **Gradient Boosting**: Often most accurate. Best for complex patterns and interactions.
            
            ### Tips for Best Results
            
            - Use at least 50-100 observations
            - For seasonal models, have at least 2x seasonal periods (e.g., 24 months for monthly data)
            - Try multiple models and compare - no single model is always best
            - Check for outliers in your data preview
            - Start with 20% test size and 10-12 step forecast horizon
            """)
        
        # Data format examples
        with st.expander("Supported Data Formats", expanded=False):
            st.markdown("""
            ### Supported Date Formats
            
            The app automatically detects common date formats:
            - `YYYY-MM-DD` (e.g., 2020-01-15)
            - `MM/DD/YYYY` (e.g., 01/15/2020)
            - `DD-MM-YYYY` (e.g., 15-01-2020)
            - `YYYY/MM/DD` (e.g., 2020/01/15)
            - Month names (e.g., Jan 2020, January 2020)
            
            ### Data Requirements
            
            - Minimum: 50 observations recommended
            - At least one date/time column
            - At least one numeric column (your target variable)
            - Regular time intervals (daily, weekly, monthly, etc.)
            
            ### Common Issues
            
            **Missing Values**: The app can handle these with 4 different strategies
            
            **Irregular Dates**: Make sure dates are evenly spaced (no random gaps)
            
            **Multiple Columns**: The app will let you select which numeric column to forecast
            
            **Large Files**: Files up to 200MB are supported
            """)

if __name__ == "__main__":
    main()