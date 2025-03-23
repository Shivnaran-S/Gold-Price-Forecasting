# Golden Time Machine: Predicting Gold Prices with ARIMA, LSTM, and Prophet Models

[Streamlit Cloud Deployment Link - Demo of Golden Time Machine](https://golden-time-machine.streamlit.app/)

## Overview

The **Golden Time Machine** is a sophisticated tool designed to predict future gold prices using historical and real-time data. Leveraging advanced machine learning models such as ARIMA, LSTM, and Prophet, this project provides actionable insights to help users make informed decisions about when to buy gold. The system is deployed on **Streamlit Cloud**, offering a seamless user experience.

## Key Features

- **Real-Time Data Extraction**: Utilizes **Selenium** and **BeautifulSoup** to scrape real-time gold prices for selected Indian cities.
- **Data Storage**: Stores the extracted data in an **SQLite** database for efficient retrieval and processing.
- **Advanced Predictive Models**: Employs **ARIMA**, **LSTM**, and **Prophet** models to forecast future gold prices.
- **User-Friendly Interface**: Built with **Streamlit**, allowing users to interact with the system effortlessly.
- **Optimal Purchase Recommendation**: Analyzes the data to recommend the best day to buy gold, minimizing potential losses.

## How It Works

1. **User Input**: Users select an Indian city from a dropdown menu.
2. **Data Extraction**: The system scrapes real-time gold prices for the selected city and updates the SQLite database.
3. **Data Processing**: The stored data is retrieved, processed, and analyzed.
4. **Model Prediction**: The ARIMA, LSTM, and Prophet models are applied to predict future gold prices.
5. **Recommendation**: The system recommends the optimal day to purchase gold based on the analysis.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web application.
- **Selenium & BeautifulSoup**: For web scraping real-time gold prices.
- **SQLite**: For data storage and retrieval.
- **Pandas & Matplotlib**: For data manipulation and visualization.
- **ARIMA, LSTM, Prophet**: For time series forecasting.

## Usage

### Step 1: Select a City
- Open the application and choose an Indian city from the dropdown menu.

### Step 2: Confirm Selection
- Click the **"Confirm Selection"** button to initiate data extraction and processing.

### Step 3: Navigate Through Pages
- **Predictor Page**: 
  - Select a date range for which you want to predict gold prices.
  - The system will analyze the data and recommend the best day to buy gold within the specified range.
- **Analysis Page**:
  - View detailed visualizations of historical gold prices.
  - Explore model performance metrics for ARIMA, LSTM, and Prophet.
  - Understand the data preprocessing steps and insights derived from the analysis.

### Step 4: View Results
- The system will display:
  - Forecasted gold prices for the selected date range.
  - A recommendation for the optimal day to purchase gold.
  - Visualizations of historical trends and model predictions.

## Results

The system provides a detailed analysis of gold prices, including:

- **Historical Price Trends**: Visualizations of past gold prices.
- **Model Performance**: Evaluation metrics for ARIMA, LSTM, and Prophet models.
- **Optimal Purchase Day**: A recommended day to buy gold based on the forecasted prices.

## Future Enhancements

- **Expand City Coverage**: Include more cities and regions.
- **Enhance Model Accuracy**: Incorporate additional features and improve model performance.
- **User Authentication**: Add user accounts to save preferences and history.

## Contact

For any inquiries, please reach out to [s.shivnaran@gmail.com](s.shivnaran@gmail.com).

---

**Golden Time Machine** is your go-to tool for making informed gold purchase decisions. Try it out today and experience the future of gold price prediction!
