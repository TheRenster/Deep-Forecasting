# 📈 Time Series Forecasting Application

A comprehensive Streamlit web application for univariate time series forecasting using both econometric and machine learning models.

## 🎯 Project Overview

This application allows users to upload time series data, apply various forecasting models, and compare their performance through interactive visualizations. Perfect for portfolio projects and practical forecasting tasks.

## ✨ Features

### Core Functionality
- **CSV File Upload**: Easy drag-and-drop interface
- **Column Selection**: Choose date/time and target variables
- **Missing Value Handling**: Multiple strategies (drop, fill, interpolate)
- **Data Filtering**: Filter by date range or last N periods
- **Train/Test Split**: Configurable holdout validation
- **Multiple Models**: Compare econometric and ML approaches
- **Interactive Visualizations**: Dynamic plots and comparisons
- **Performance Metrics**: RMSE, MAE, and MAPE

### Forecasting Models

#### Econometric Models (statsmodels)
1. **Holt-Winters Exponential Smoothing**
   - Captures trend and seasonality
   - Additive seasonal and trend components
   - Configurable seasonal periods

2. **SARIMA** (Seasonal ARIMA)
   - Handles seasonal patterns
   - Configurable order and seasonal order
   - Robust for complex patterns

#### Machine Learning Models (scikit-learn)
1. **Random Forest Regressor**
   - Ensemble of decision trees
   - Feature engineering with lags and rolling statistics
   - Resistant to overfitting

2. **Gradient Boosting Regressor**
   - Sequential ensemble learning
   - Strong predictive performance
   - Handles non-linear relationships

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ts-forecasting-app.git
cd ts-forecasting-app
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 How to Use

### Step 1: Upload Data
- Click "Browse files" or drag and drop a CSV file
- Your CSV should contain:
  - A date/time column (any format pandas can parse)
  - At least one numeric target variable

### Step 2: Configure Data
1. **Select Columns**: Choose date and target variable
2. **Handle Missing Values**: Pick a strategy if missing data exists
3. **Filter Data** (optional): Focus on specific time periods

### Step 3: Set Forecast Parameters
- **Test Size**: Percentage of data for testing (10-40%)
- **Forecast Horizon**: Number of future periods to predict
- **Seasonal Periods**: For econometric models (e.g., 12 for monthly data)
- **Number of Lags**: For ML models (3-24 previous values)

### Step 4: Select Models
- Choose at least one econometric model (Holt-Winters or SARIMA)
- Choose at least one ML model (Random Forest or Gradient Boosting)

### Step 5: Train and Evaluate
- Click "Train Models and Generate Forecasts"
- Review metrics table to compare performance
- Explore visualizations:
  - Individual model forecasts
  - Side-by-side comparison
  - Metrics bar charts

## 📈 Data Format

Your CSV should follow this structure:

```csv
Date,Sales
2020-01-01,1200
2020-02-01,1350
2020-03-01,1280
2020-04-01,1420
...
```

**Requirements:**
- Date column: Any format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- Target column: Numeric values only
- Recommended: At least 50-100 observations for meaningful forecasts

## 🧮 Evaluation Metrics

### RMSE (Root Mean Squared Error)
- Measures average prediction error
- **Lower is better**
- Sensitive to outliers
- Same units as target variable

### MAE (Mean Absolute Error)
- Average absolute difference between predictions and actual values
- **Lower is better**
- More robust to outliers than RMSE
- Same units as target variable

### MAPE (Mean Absolute Percentage Error)
- Percentage-based error metric
- **Lower is better**
- Easy to interpret (e.g., 5% error)
- Scale-independent

## 🎨 Feature Engineering (ML Models)

The application automatically creates features for ML models:

### Lag Features
- `lag_1` to `lag_n`: Previous n time periods
- Captures temporal dependencies

### Rolling Statistics
- `rolling_mean_3`: 3-period moving average
- `rolling_mean_6`: 6-period moving average
- `rolling_std_3`: 3-period rolling standard deviation

These features allow ML models to learn temporal patterns without explicit time series formulation.

## 🔧 Troubleshooting

### Common Issues

**"Error parsing date column"**
- Ensure dates are in a consistent format
- Try different date column selections
- Check for non-date values in the column

**"Model training failed"**
- Ensure sufficient data (at least 2x seasonal periods for seasonal models)
- Check for extreme values or outliers
- Try adjusting seasonal periods parameter

**"Not enough data for lag features"**
- Reduce number of lags
- Use more training data
- Remove data filtering

## 🎓 Model Selection Guide

### When to use Holt-Winters
- Clear trend and seasonal patterns
- Regular seasonality (monthly, quarterly)
- Smooth, continuous data

### When to use SARIMA
- Complex seasonal patterns
- Need for differencing (non-stationary data)
- Multiple seasonal periods

### When to use Random Forest
- Non-linear relationships
- Irregular patterns
- Large datasets
- Robust to outliers

### When to use Gradient Boosting
- Need highest accuracy
- Sufficient computational resources
- Complex interactions between lags

## 📦 Project Structure

```
ts-forecasting-app/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── sample_data/          # (Optional) Sample datasets
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in and click "New app"
4. Select your repository and branch
5. Set main file to `app.py`
6. Click "Deploy"

### Alternative: Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Econometric models from [statsmodels](https://www.statsmodels.org/)
- ML models from [scikit-learn](https://scikit-learn.org/)

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/ts-forecasting-app](https://github.com/yourusername/ts-forecasting-app)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
