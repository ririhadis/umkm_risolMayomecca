# SmartStock AI — Demand Forecasting & Production Decision Support

SmartStock AI is an end-to-end **demand forecasting and production decision-support application** developed for **Risol Mayo Mecca**, a small food business.

The application uses a **Temporal Fusion Transformer (TFT)** to forecast next-day sales for multiple risol variants and translates the forecasts into actionable recommendations for:

* Daily risol production quantities
* Raw-material requirements
* Best-selling and slow-moving products
* Next-day production planning
* Automated Telegram notifications

The system is designed to help reduce overproduction, minimize stock shortages, and support more data-driven daily production decisions.

## 🚀 Live Application

**Streamlit App:**
https://umkmrisolmayomecca-smartstock.streamlit.app/

---

## 🎯 Business Problem

Daily production planning for food businesses can be challenging because demand varies depending on:

* Day of the week
* Historical sales patterns
* Holidays and special events
* Product popularity
* Store operating conditions
* Previous sales performance

Producing too much can increase unsold inventory and waste, while producing too little can result in stock shortages and missed sales opportunities.

SmartStock AI addresses this problem by forecasting the expected demand for the following day and converting the forecast into practical production and raw-material recommendations.

---

## 💡 Solution

SmartStock AI follows the workflow:

```text
Daily Sales Input
       ↓
Google Sheets
       ↓
GCP Authentication & Data Connection
       ↓
Data Processing
       ↓
Temporal Fusion Transformer (TFT)
       ↓
Next-Day Demand Forecast
       ↓
Production Recommendation
       ↓
Raw-Material Requirement Calculation
       ↓
Streamlit Dashboard
       ↓
Telegram Notification
```

The system creates a continuous feedback loop where actual sales data can be stored and used as historical information for subsequent forecasting.

```text
Actual Sales
     ↓
Google Sheets
     ↓
Forecasting Model
     ↓
Next-Day Recommendation
     ↓
Actual Sales of Next Day
     ↓
Google Sheets
     ↺
```

---

## 🧠 Machine Learning Approach

### Temporal Fusion Transformer (TFT)

The forecasting model used in this project is the **Temporal Fusion Transformer (TFT)**.

TFT was selected because the forecasting problem contains:

* Temporal sales patterns
* Multiple product variants
* Historical sales information
* Known future calendar information
* Holiday effects
* Valid zero-sales observations

The model forecasts demand for the following risol variants:

* Ayam
* Udang
* Keju
* Sosis
* Telur

### Why TFT?

One of the important characteristics of the dataset is the presence of **zero-sales observations**.

A value of `0` does not necessarily represent missing data. On certain days, the store may be closed, meaning that zero sales are a valid business observation.

Instead of automatically removing these observations, they are retained as part of the time-series data.

The model also uses known future information such as calendar and holiday features, allowing the forecasting system to consider expected changes in demand associated with specific dates.

---

## 📅 Future & Calendar Features

The forecasting process incorporates calendar-related information that is known before the prediction date, including:

* Date
* Day of week
* Holiday information
* Special calendar events

These features help the model distinguish normal operating days from periods where sales behavior may differ.

For example, a store-closure day may legitimately produce:

```text
Sales = 0
```

while a holiday or special event may produce a different demand pattern.

---

## 📊 Dataset

The application uses approximately **224 days of historical sales data**.

The historical data contains daily sales information for multiple risol variants.

The main target variables include:

```text
Ayam
Udang
Keju
Sosis
Telur
```

Daily operational data is stored and retrieved through **Google Sheets**, with the Streamlit application connected using **GCP authentication**.

This allows the application to function as an operational forecasting system rather than a static machine-learning demo.

---

## 🔄 Data Flow

### 1. Daily Sales Input

Users can input actual sales for each risol variant:

* Risol Ayam
* Risol Udang
* Risol Keju
* Risol Sosis
* Risol Telur

The application calculates the total units sold and remaining sales/inventory information.

The data can then be saved for use as historical information.

### 2. Data Storage

Sales data is stored in **Google Sheets**.

The Streamlit application connects to Google Sheets through **GCP authentication** for data retrieval and storage.

### 3. Forecasting

The historical data is processed and passed to the TFT forecasting model.

The model generates next-day demand predictions for each risol variant.

### 4. Production Recommendation

The forecast is translated into recommended production quantities for the following day.

Example:

```text
Risol Ayam   → 1,298 pcs
Risol Udang  →   212 pcs
Risol Keju   →   148 pcs
Risol Telur  →   145 pcs
Risol Sosis  →   140 pcs
```

*Example values shown above are based on application output and may change as new data becomes available.*

### 5. Raw-Material Calculation

The recommended production quantities are converted into estimated raw-material requirements.

Examples include:

```text
Chicken
Flour
Butter
Mayonnaise
Breadcrumbs
Potato & Carrot
```

This creates a direct connection between demand forecasting and production planning.

### 6. Telegram Notification

The final recommendation is delivered through Telegram.

The notification contains:

* Recommended number of risol to produce for the next day
* Recommendation for each menu/variant
* Estimated raw-material requirements for the next day

This allows the business owner to receive actionable information without having to continuously monitor the dashboard.

---

## 📈 Business Analytics

In addition to forecasting, SmartStock AI provides several business insights.

### Historical Sales Trend

The application visualizes recent sales trends to help users understand changes in demand over time.

### Best Seller Analysis

Identifies products with the highest recent sales performance.

### Slow Mover Analysis

Identifies products with relatively lower recent sales performance.

These analyses can support decisions about production allocation and inventory planning.

---

## 🚀 Live Dashboard & Smart Stock System

This application is used to monitor the stock and transactions of UMKM Risol Mayo Mecca in real-time. 

To view stock status, daily metrics, and the latest data analytics, please access the application directly via the following link:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://umkmrisolmayomecca-smartstock.streamlit.app/)

> 💡 **Note:** The metric figures and stock levels in this repository's documentation are static/example data. All live and updated data can be viewed directly on the application link above.

### Metric Interpretation

**MAE (Mean Absolute Error)**
Measures the average absolute difference between predicted and actual sales.

**RMSE (Root Mean Squared Error)**
Penalizes larger forecasting errors more strongly than MAE.

**WMAPE (Weighted Mean Absolute Percentage Error)**
Measures forecasting error relative to the total actual demand and provides an interpretable percentage-based metric.

The achieved **low WMAPE** indicates relatively low forecasting error compared with the total sales volume in the evaluated data.

---

## 🏪 Production Decision Support

The main purpose of the model is not only to generate predictions.

The prediction is transformed into an operational decision:

```text
Forecasted Demand
       ↓
Recommended Production
       ↓
Required Raw Materials
       ↓
Daily Production Planning
```

This makes SmartStock AI a **decision-support system**, rather than a standalone forecasting model.

---

## 📱 Streamlit Dashboard

The application provides an interactive interface containing:

* Daily sales input
* Total sales and remaining inventory information
* Historical sales trends
* Demand forecasting
* Production recommendations
* Best seller analysis
* Slow mover analysis
* Raw-material requirements
* Model performance metrics
* Telegram notification integration

---

## 🛠️ Technology Stack

### Programming & Data

* Python
* Pandas
* NumPy

### Machine Learning

* Temporal Fusion Transformer (TFT)
* Time-series forecasting
* Feature engineering
* Model evaluation

### Data Storage & Cloud

* Google Sheets
* Google Cloud Platform (GCP)
* GCP authentication

### Application & Deployment

* Streamlit
* Streamlit Cloud

### Automation & Notification

* Telegram Bot API

---

## 🏗️ System Architecture

```text
                     ┌─────────────────────┐
                     │    Google Sheets    │
                     │                     │
                     │ Historical Sales    │
                     │ Daily Sales Data    │
                     └──────────┬──────────┘
                                │
                         GCP Authentication
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Streamlit App     │
                     │                     │
                     │ Data Retrieval      │
                     │ Data Input          │
                     │ Data Processing     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Feature Engineering │
                     │                     │
                     │ Calendar Features   │
                     │ Holiday Features    │
                     │ Historical Sales    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │       TFT Model     │
                     │                     │
                     │ Demand Forecasting  │
                     └──────────┬──────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ Next-Day Demand Forecast     │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
       Production Recommendation       Raw-Material Requirement
                 │                              │
                 └──────────────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             Streamlit Dashboard     Telegram Notification
```

---

## 🔐 Data & Security

The application uses external services such as Google Sheets, GCP authentication, and Telegram.

Sensitive credentials and API tokens should **never be committed to the repository**.

Recommended practice:

* Store credentials using environment variables or Streamlit Secrets
* Do not hard-code Telegram Bot Tokens
* Do not commit GCP service-account credentials
* Do not expose private API credentials in screenshots or documentation
* Rotate/revoke credentials immediately if they are accidentally exposed

---

## 📌 Key Features

* [x] Daily sales data input
* [x] Google Sheets data storage
* [x] GCP-based authentication
* [x] Historical sales analysis
* [x] Multi-product demand forecasting
* [x] Temporal Fusion Transformer (TFT)
* [x] Holiday and calendar features
* [x] Handling of valid zero-sales observations
* [x] Next-day production recommendation
* [x] Raw-material requirement estimation
* [x] Best seller analysis
* [x] Slow mover analysis
* [x] Model performance monitoring
* [x] Streamlit deployment
* [x] Telegram recommendation notification

---

## 📊 Business Impact

SmartStock AI is designed to help a small food business:

* Make production planning more data-driven
* Reduce unnecessary overproduction
* Reduce the risk of stock shortages
* Estimate raw-material requirements before production
* Identify high- and low-performing products
* Receive next-day recommendations automatically
* Reduce dependence on manual demand estimation

The core principle is:

> **Forecast → Decide → Produce**

rather than relying solely on intuition or manual estimation.

---

## 👩‍💻 Role & Contribution

**Role: Data Scientist / Machine Learning Developer**

Key contributions include:

* Designed the demand forecasting workflow
* Prepared and processed historical sales data
* Developed forecasting features
* Selected and implemented the TFT forecasting approach
* Incorporated calendar and holiday information
* Handled valid zero-sales observations caused by store closures
* Evaluated forecasting performance using MAE, RMSE, and WMAPE
* Developed logic to translate forecasts into production recommendations
* Developed raw-material requirement calculations
* Integrated the forecasting workflow into Streamlit
* Connected the application to Google Sheets through GCP authentication
* Integrated Telegram for automated operational recommendations

---

## 🎓 Data Science Concepts Demonstrated

This project demonstrates practical application of:

* Time-series forecasting
* Temporal Fusion Transformer
* Feature engineering
* Calendar feature engineering
* Holiday/event feature engineering
* Multi-product forecasting
* Model evaluation
* Business analytics
* Descriptive analytics
* Predictive analytics
* Prescriptive analytics
* Decision-support systems
* Data-to-business workflow design

---

## 🔮 Future Improvements

Potential improvements include:

* Automated model retraining when new sales data becomes available
* Forecast confidence intervals
* More advanced inventory optimization
* Stockout and overproduction cost optimization
* Automated model performance monitoring
* Experiment tracking
* Model versioning
* Database integration for larger-scale data
* Automated data validation
* Forecast drift monitoring
* More granular holiday and event effects
* Multi-step forecasting for weekly production planning

---

## 🌐 Live Demo

Try the deployed application:

https://umkmrisolmayomecca-smartstock.streamlit.app/

---

## 📌 Project Summary

**SmartStock AI transforms historical sales data into actionable next-day production decisions.**

The system combines:

**Google Sheets + GCP + Time-Series Forecasting + TFT + Streamlit + Telegram**

to create an end-to-end demand forecasting and production decision-support solution for a real-world small food business.

The project demonstrates how machine learning can move beyond prediction accuracy and become part of an operational workflow:

```text
DATA
  ↓
ANALYSIS
  ↓
FORECAST
  ↓
DECISION
  ↓
ACTION
```

---

## ⭐ Portfolio Highlight

> **An end-to-end demand forecasting and production decision-support system using Temporal Fusion Transformer (TFT), historical sales data, calendar/holiday features, Google Sheets, GCP authentication, Streamlit, and Telegram automation to generate next-day production and raw-material recommendations.**