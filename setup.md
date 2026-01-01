# DCF Valuation Model - Setup Guide

A professional Discounted Cash Flow (DCF) valuation application built with Streamlit.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AIAydin/DCF-Model.git
cd DCF-Model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.28.0 | Web application framework |
| yfinance | ≥0.2.31 | Yahoo Finance API for stock data |
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computations |
| plotly | ≥5.18.0 | Interactive charts |

## 🎯 Features

- **Real-time Stock Data**: Fetches live financial data from Yahoo Finance
- **DCF Valuation**: Calculates intrinsic value using discounted cash flow methodology
- **Sensitivity Analysis**: Interactive heatmap showing value under various assumptions
- **Beautiful UI**: Modern, professional design with responsive charts

## 💡 Usage

1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)
2. Adjust growth rate assumptions using the sidebar sliders
3. Click "Analyze Stock" to run the DCF valuation
4. Review the intrinsic value and sensitivity analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
