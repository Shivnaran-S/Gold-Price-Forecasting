import streamlit as st
import sqlite3
import os 

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")
chrome_options.binary_location = "/usr/bin/chromium"  # Adjust path if needed

import pandas as pd
import matplotlib.pyplot as plt
import builtins
from datetime import datetime

# Save the original print function
Print = builtins.print

# Save the original plt.show
Plot = plt.show

# Override the print function
def print(*args, **kwargs):
    st.write(*args, **kwargs)

# Override plt.show() with st.pyplot()
def custom_show(*args, **kwargs):
    st.pyplot(plt)

# Override plt.show
plt.show = custom_show

DB_NAME = "gold_rates.db"

city = "coimbatore"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_rates (
            city TEXT,
            date DATE,
            morning_price REAL,
            evening_price REAL,
            PRIMARY KEY (city, date)
        )
    """)
    conn.commit()
    conn.close()
    
def fetch_data(city):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    query = f"""
        SELECT date, morning_price, evening_price
        FROM gold_rates
        WHERE city = ?
    """
    cursor.execute(query, (city,))
    data = cursor.fetchall()
    

    if not data:
        data = init_data(city)
        df = data.copy()

    else:
        data, df = update_data(city, data) # data will be having the original data from db with updated data while 
                                     # df will have the last few rows of data which are not in db and the data which is webscarped now and it must be added to the db

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['Morning'] = df['Morning'].astype(float)
    df['Evening'] = df['Evening'].astype(float)

    for index, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO gold_rates (city, date, morning_price, evening_price)
                VALUES (?, ?, ?, ?)
            """, (city, row['Date'], row['Morning'], row['Evening']))

    conn.commit()
    conn.close()
    return data

def init_data(city):
    st.write("Please wait! The data is beeing fetched.")
    st.write("\n")
    data1 = fetch_all_data_1(city)

    st.write("Really sorry for making you to wait!") 
    st.write("Please wait for some more time! The data is beeing fetched.")    
    st.write("\n")   
    data2 = fetch_all_data_2(city)
    
    st.write("Done! Fetched data. Thank you so much for your patience!")
    #data3 = pd.concat([data1, data2], ignore_index=True)
    #data4 = fetch_month_data()

    data = pd.concat([data1, data2], ignore_index=True)
    return data

def update_data(city, data):
    data = pd.DataFrame(data, columns=['Date', 'Morning', 'Evening'])
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%d-%b-%y')

    import datetime as dt
    last_date = data.iloc[-1,0]
    last_date = dt.datetime.strptime(last_date, '%d-%b-%y')
    s_day = int( last_date.strftime('%d') )
    s_month = int( last_date.strftime('%m') )
    s_year = int( last_date.strftime('%Y') )

    df = fetch_all_data_2(city, s_day+1,s_month,s_year)
    data = pd.concat([data, df], ignore_index=True)
    return data,df

def fetch_monthly_data_1(city, month, year):
    url = f'https://www.indgold.com/{city}-gold-rate-{month}-{year}.htm'
    
    service = Service("/usr/bin/chromedriver")   
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)
    html_content = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html_content, 'html.parser')

    tables = soup.find_all('table')
    if len(tables) >= 2:
        target_table = tables[1]
        rows = target_table.find_all('tr')
        data = []

        for row in rows[1:]:
            cols = row.find_all('td')
            data.append([col.get_text(strip=True) for col in cols])

        month_df = pd.DataFrame(data, columns=['Date', 'Morning', 'Evening'])
        return month_df
    else:
        return pd.DataFrame(columns=['Date', 'Morning', 'Evening'])
    
def fetch_all_data_1(city):
    start_year, end_year = 2021, 2023
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july','august', 'september', 'october', 'november', 'december']

    all_data = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        for month in months:
            if year == 2021 and month not in ['august', 'september', 'october', 'november', 'december']:
              continue
            if year == 2023 and month == 'august':
              break

            month_data = fetch_monthly_data_1(city, month, year)

            all_data = pd.concat([all_data, month_data], ignore_index=True)
            
    return all_data
    #all_data.to_csv('gold_rate_data_aug2021_jul2023.csv', index=False)

def fetch_monthly_data_2(city, month, year):
    url = f'https://www.indgold.com/{city}-gold-rate-{month}-{year}.htm'
    # Set up the WebDriver
    service = Service()  # chromedriver_autoinstaller handles the path automatically
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Fetch the webpage
    driver.get(url)
    html_content = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html_content, 'html.parser')

    table_div = soup.find('div', id='table')
    if table_div:
        rows = table_div.find_all('tr')
        
        data = []

        for row in rows[1:]:
            cols = row.find_all('td')
            data.append([col.get_text(strip=True) for col in cols])

        month_df = pd.DataFrame(data, columns=['Date', 'Morning', 'Evening'])
        return month_df
    else:
        return pd.DataFrame(columns=['Date', 'Morning', 'Evening'])

def fetch_all_data_2(city, s_day=1,s_month=8,s_year=2023): #Starting from which date we need to fetch data
    s_month-=1
    s_day-=1 #for indexing
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    today = datetime.today()
    t_day = today.day
    t_month = today.month-1
    t_year = today.year

    all_data = pd.DataFrame()

    for year in range(s_year,t_year+1):
        for month in months:
            #if year == 2023 and month not in months[months.index("august"):]:
            #    continue
            if year == s_year and month not in months[s_month:]:
                continue
            if year == t_year and month == months[t_month]:
                break
            
            month_data = fetch_monthly_data_2(city, month, year)
            all_data = pd.concat([all_data, month_data], ignore_index=True)

    if s_month == t_month and s_year==t_year:
        df = fetch_month_data(city, s_day)
    else:
        df = fetch_month_data(city)
        all_data = all_data.iloc[s_day:]
    all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data
    #combined_data.to_csv('gold_rate_data_aug2021_dec2024.csv', index=False)

def fetch_month_data(city, s_day=0):
    if city in ['mumbai','delhi','chennai','bangalore','hyderabad','cochin']:
        url = f'https://www.indgold.com/{city}-gold-rates.htm'
    else:
        url = f'https://www.indgold.com/{city}-gold-rate.htm'
    # Set up the WebDriver
    service = Service()  # chromedriver_autoinstaller handles the path automatically
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Fetch the webpage
    driver.get(url)
    html_content = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html_content, 'html.parser')

    table_div = soup.find('div', id='table')
    if table_div:
        rows = table_div.find_all('tr')
        data = []
        for row in rows[1:]:
            cols = row.find_all('td')
            row_data = [col.get_text(strip=True) for col in cols]
            data.append(row_data)

        df = pd.DataFrame(data, columns=['Date', 'Morning', 'Evening'])
        return df.iloc[s_day:]
    else:
        #print("No <div> found with id='table'.")
        return pd.DataFrame(columns=['Date', 'Morning', 'Evening'])

def final_model(data, from_date, to_date):
    from prophet import Prophet
    
    data_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna() # train_data.iloc[:,-1]
    data_series.index.freq = pd.infer_freq(data_series.index) # 'D'
    
    df = data_series.reset_index()        
    df.columns = ['ds','y']
    
    model = Prophet()
    model.fit(df)
    
    # Create a future dataframe for prediction
    # Create a future dataframe for prediction
    future = model.make_future_dataframe(periods=(to_date - df['ds'].max().date()).days)

    # Predict the forecast
    forecast = model.predict(future)

    # Filter the forecasted data for the specified range
    filtered_forecast = forecast[(forecast['ds'] >= pd.Timestamp(from_date)) & (forecast['ds'] <= pd.Timestamp(to_date))]

    # Find the date with the minimum forecasted value within the range
    min_forecast_row = filtered_forecast.loc[filtered_forecast['yhat'].idxmin()]  # Get the row with the minimum value
    min_date = min_forecast_row['ds']
    min_value = min_forecast_row['yhat']
    return min_date, min_value
              
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Predictor", "Analysis"])
    #data = pd.DataFrame(columns=['Date', 'Morning', 'Evening'])
    if "is_predictor_done" not in st.session_state:
        st.session_state.is_predictor_done = False  # Flag to track if Predictor step is completed
    if "is_analysis_done" not in st.session_state:
        st.session_state.is_analysis_done = False
    global city
    city = "coimbatore"
    if page=="Predictor":
        if not st.session_state.is_predictor_done:
            st.title("GOLDEN TIME MACHINE")
        
            cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Ahmedabad","Surat","Pune","Mysore","Mangalore","Madurai","Tiruchirappalli","Coimbatore","Salem","Cochin","Kozhikode","Thiruvananthapuram","Thrissur"]
            cities = sorted(cities)
            
            default_city = "Coimbatore"
            city = st.selectbox("Select a city from where you want to buy gold: ", cities, index=cities.index(default_city))
            if st.button("Confirm Selection"):
                print()
            else:
                exit()
            city = city.lower()
            
            init_db()
            st.session_state.is_predictor_done = True
        if not st.session_state.is_analysis_done:
            st.error("Let the Analysis get completed first")
        else:
            today_date = datetime.today().date()
            from_date = st.date_input("Select from which date you are planning to buy gold:", value=today_date,min_value=today_date)
            to_date = st.date_input("Select before which date you are planning to buy gold, starting from the above date:", value=from_date,min_value=from_date)

            data = fetch_data(city)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            data.set_index('Date', inplace=True)
            data.index = pd.to_datetime(data.index)
        
            # Optionally, infer frequency if it's consistent (daily, monthly, etc.)
            if data.index.is_unique:
                data = data.asfreq('D') 
            data['Evening_Differenced_1'] = [None] * len(data)
            for i in range(1, len(data)):
                data.iloc[i,-1] = data.iloc[i,1] - data.iloc[i-1,1]
                
            day, price = final_model(data, from_date, to_date)
            print(f"Finally we are done. Buy gold on {day}. This will minimize your loss eventhough it couldn't maximize your profit")
    if page == "Analysis":
        if not st.session_state.is_predictor_done:
            st.error("Please complete the Predictor step first!")
        else:
            '''DATA'''
            #Load the dataset and make Date as index of the dataframe
            #data = pd.read_csv("gold_rate_data_aug2021_jan2025.csv")
            data = fetch_data(city)
            df = data.copy()
            
            #print(data)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            #data['Date'] = pd.to_datetime(data['Date'],format='%d-%b-%y').dt.date
            data.set_index('Date', inplace=True)
        
            print("THE FOLLOWING IS THE DATASET")
            print(data)
            
            #Prepare the train and test data set
            no_of_rows = len(data)
            train_data = data.iloc[:no_of_rows-30,:]
            test_data = data.iloc[no_of_rows-30:,:]
            print("Length of the train data is : ",len(train_data))
            print("Length of the test data is : ",len(test_data))
        
            '''DATA PREPROCESSING'''
            #Find if any data point is missing
            print("Total null values in the dataset: ",data.isnull().sum().sum())
        
            #Find if there is any duplicate row
            #print(data.duplicated().sum())  # In data the date is set as index, so eventhough when the timestamp differs all those will be considered as duplicate data
            print("Number of duplicate values in the data set is : ",df.duplicated().sum()) # Here Date is a separate column, so different timestamp's similar rates are not considered as duplicates
            
            # Convert 'Morning' and 'Evening' columns to numeric
            data['Morning'] = pd.to_numeric(data['Morning'], errors='coerce').dropna()
            data['Evening'] = pd.to_numeric(data['Evening'], errors='coerce').dropna()
            #Find if there are any outliers
            Q1 = data[['Morning', 'Evening']].quantile(0.25)
            Q3 = data[['Morning', 'Evening']].quantile(0.75)
            IQR = Q3 - Q1
        
            outliers = ((data[['Morning', 'Evening']] < (Q1 - 1.5 * IQR)) |
                        (data[['Morning', 'Evening']] > (Q3 + 1.5 * IQR))).sum()
            print("Outliers : ",outliers)
        
            #Outlier plot for Morning data
            plt.figure(figsize=(8, 4))
            plt.boxplot(data['Morning'])
            plt.title('Boxplot for Morning Prices')
            plt.ylabel('Price')
        
            morning_stats = data['Morning'].describe()
            max_value = morning_stats['max']
            min_value = morning_stats['min']
            median_value = morning_stats['50%']
        
            plt.scatter(1, max_value, label=f'Max: {max_value:.2f}', color='orange')
            plt.scatter(1, min_value, label=f'Min: {min_value:.2f}', color='blue')
            plt.scatter(1, median_value, label=f'Median: {median_value:.2f}', color='green')
        
            plt.text(1.1, max_value, f'{max_value:.2f}', color='orange')
            plt.text(1.1, min_value, f'{min_value:.2f}', color='blue')
            plt.text(1.1, median_value, f'{median_value:.2f}', color='green')
        
            plt.legend()
            plt.show()
        
            #Outlier plot for Evening data
            plt.figure(figsize=(8, 4))
            plt.boxplot(data['Evening'])
            plt.title('Boxplot for Evening Prices')
            plt.ylabel('Price')
        
            evening_stats = data['Evening'].describe()
            max_value = evening_stats['max']
            min_value = evening_stats['min']
            median_value = evening_stats['50%']
        
            plt.scatter(1, max_value, label=f'Max: {max_value:.2f}', color='red')
            plt.scatter(1, min_value, label=f'Min: {min_value:.2f}', color='yellow')
            plt.scatter(1, median_value, label=f'Median: {median_value:.2f}', color='green')
        
            plt.text(1.1, max_value, f'{max_value:.2f}', color='red')
            plt.text(1.1, min_value, f'{min_value:.2f}', color='yellow')
            plt.text(1.1, median_value, f'{median_value:.2f}', color='green')
        
            plt.legend()
            plt.show()
        
            '''NORMALIZATION OF THE DATA - ANALYSIS ONLY, DID NOT USE IT'''
            normalized_data = data.copy()
        
            min_morning = normalized_data['Morning'].min()
            max_morning = normalized_data['Morning'].max()
            min_evening = normalized_data['Evening'].min()
            max_evening = normalized_data['Evening'].max()
        
            normalized_data['Morning'] = (normalized_data['Morning'] - min_morning) / (max_morning - min_morning)
            normalized_data['Evening'] = (normalized_data['Evening'] - min_evening) / (max_evening - min_evening)
            print(normalized_data)
        
            '''DATA ANALYTICS AND VISUALISATION'''
            #Morning
            plt.figure(figsize=(15, 7))
            plt.plot(data['Morning'])
            plt.title('Gold Price - Morning')
            plt.grid(True)
            plt.show()
        
            #Evening
            plt.figure(figsize=(15, 7))
            plt.plot(data['Evening'])
            plt.title('Gold Price - Evening')
            plt.grid(True)
            plt.show()
        
            #Both
            data.plot(figsize=(15,7))
            plt.xlabel("Date")
            plt.ylabel("Gold Price")
            plt.title("Date v/s Gold Price")
            plt.show()
        
            def year_wise_plot(start_date, end_date):
                start_date = pd.to_datetime(start_date, format='%d-%b-%y').date()
                end_date = pd.to_datetime(end_date, format='%d-%b-%y').date()
                
                filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
                # Calculate the min and max values for 'Morning' and 'Evening' columns
                min_val = min(filtered_data['Morning'].min(), filtered_data['Evening'].min())
                max_val = max(filtered_data['Morning'].max(), filtered_data['Evening'].max())
        
                data.plot(xlim=[start_date,end_date],ylim=[min_val,max_val],figsize=(15,4))
                plt.xlabel("Date")
                plt.ylabel("Gold Price")
                plt.title(f"Time series plot from {start_date} to {end_date}")
                plt.show()
        
            #21-22
            year_wise_plot('1-Aug-21', '31-Jul-22')
            
            #22-23
            year_wise_plot('1-Aug-22', '31-Jul-23')
            
            #23-24
            year_wise_plot('1-Aug-23', '31-Jul-24')
        
            #24-25
            x_max = data.index.max()
            
            start_date = pd.to_datetime('1-Aug-24', format='%d-%b-%y').date()
            filtered_data = data[(data.index >= start_date)]
        
            min_val = min(filtered_data['Morning'].min(), filtered_data['Evening'].min())
            max_val = max(filtered_data['Morning'].max(), filtered_data['Evening'].max())
            
            data.plot(xlim=[start_date,x_max],ylim=[min_val,max_val],figsize=(15,4))
            plt.xlabel("Date")
            plt.ylabel("Gold Price")
            plt.title("Gold Price for the year 2024-25")
            plt.show()
            
            #Rolling mean and standard deviation
            rolling_window = 30  # size of the rolling window is 30 as there are 30 days in a month
        
            normalized_data['Morning_RollingMean'] = normalized_data['Morning'].rolling(window=rolling_window).mean()
            normalized_data['Morning_RollingStd'] = normalized_data['Morning'].rolling(window=rolling_window).std()
        
            plt.figure(figsize=(14, 6))
            plt.plot(normalized_data['Morning'], label='Morning Rate')
            plt.plot(normalized_data['Morning_RollingMean'], label='30-Day Rolling Mean', color='orange')
            plt.plot(normalized_data['Morning_RollingStd'], label='30-Day Rolling Std Dev', color='green')
            plt.title("Rolling Mean & Std Deviation for Morning Rate")
            plt.xlabel("Date")
            plt.ylabel("Normalized Price")
        
            plt.legend()
            plt.show()
        
            #Statistical plots
            import statsmodels.api as sm
            data.index = pd.to_datetime(data.index)
        
            # Optionally, infer frequency if it's consistent (daily, monthly, etc.)
            if data.index.is_unique:
                data = data.asfreq('D')  # or 'M' for monthly, depending on your data
        
        
            # Y(t) = Trend(t) + Seasonal(t) + Residual(t)
            decomposition = sm.tsa.seasonal_decompose(data['Morning'], model='additive')
            fig = decomposition.plot()
            fig.set_size_inches(14, 10)
            plt.suptitle("Additive Decomposition of Morning Gold Prices: Original, Trend, Seasonal, and Residual Components", fontsize=16)
            plt.show()
        
            # Y(t) = Trend(t) × Seasonal(t) × Residual(t)
            decomposition = sm.tsa.seasonal_decompose(data['Evening'], model='multiplicative')
            fig = decomposition.plot()
            fig.set_size_inches(14, 10)
            plt.suptitle("Multiplicative Decomposition of Evening Gold Prices: Original, Trend, Seasonal, and Residual Components", fontsize=16)
            plt.show()
        
            plt.close()
        
            # EEDDAA
            print("EXPLORATORY DATA ANALYSIS - EDA")
            print("ANALYZING THE TIME SEREIS DATA FOR CHECKING STATIONARITY THROUGH STATISTICAL VISUALIZATIONS")
            #yt versus yt-1
            plt.scatter(data.iloc[:-1,1],data.iloc[1:,1]) 
            plt.title('y_t versus y_t+1')
            plt.show()
            plt.close()
            #For a stationary process that the nature of the joint probability distribution 𝑝(𝑧 𝑡, 𝑧 𝑡+𝑘) of values separated by 𝑘 intervals of time can be inferred by plotting a scatter diagram using pairs of values(𝑧 𝑡, 𝑧 𝑡+𝑘) of the time series, separated by a constant interval or lag k
            lag_k = 730 # 2 years
        
            z_t = data.iloc[:-lag_k,1]  # Excluding the last 'k' values
            z_t_k = data.iloc[lag_k:,1]  # Excluding the first 'k' values
        
            plt.figure(figsize=(8, 6))
            plt.scatter(z_t, z_t_k)
        
            plt.title(f"Scatter Plot of z_t vs z_t+{lag_k}")
            plt.xlabel("z_t")
            plt.ylabel(f"z_t+{lag_k}")
            plt.grid(True)
            plt.show()
            plt.close()
        
            #For different time lags and different sets of data
            plt.scatter(data.iloc[500:865,1],data.iloc[500+365:865+365,1])
            plt.title("Scatter Plot of y_500:865 vs y_865:1230")
            plt.xlabel("y_500:865")
            plt.ylabel("y_865:1230")
            plt.grid(True)
            plt.show()
            plt.close()
        
            plt.scatter(data.iloc[500:865,1],data.iloc[500-365:865-365,1])
            plt.title("Scatter Plot of y_500:865 vs y_135:500")
            plt.xlabel("y_500:865")
            plt.ylabel("y_135:500")
            plt.grid(True)
            plt.show()
        
            #Checking for constant mean
            print("Mean of y_t  : ",data.iloc[:-1,1].mean())
            print("Mean of y_t+1: ",data.iloc[1:,1].mean())
        
            #Plotting rolling mean and standard deviation
            rmean = data['Evening'].rolling(window=30).mean().dropna()
            rstd = data['Evening'].rolling(window=30).std().dropna()
        
            plt.figure(figsize=(14,5))
        
            plt.plot(data['Evening'], label='Original', color='black')
            plt.plot(rmean, label='Rolling Mean', color='red',)
            plt.plot(rstd, label = 'Rolling Standard Deviation', color='blue')
        
            plt.title("Rolling - mean and standard deviation")
            plt.legend(loc='best')
            plt.show()
        
            # Time series plot
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Evening'], color='blue', label='Evening Price')
            plt.title('Gold Evening Price Over Time')
            plt.xlabel('Date')
            plt.ylabel('Evening Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            plt.figure(figsize=(12, 6))
            plot_acf(data['Evening'], lags=50)
            plt.title("Autocorrelation (ACF) for Evening Rate")
            plt.show()
        
            plt.figure(figsize=(12, 6))
            plot_pacf(data['Evening'], lags=50)
            plt.title("Partial Autocorrelation (PACF) for Evening Rate")
            plt.show()
        
            plt.close()
        
            print("Found that the data is not stationary")
        
            data['Evening_Differenced_1'] = [None] * len(data)
            for i in range(1, len(data)):
                data.iloc[i,-1] = data.iloc[i,1] - data.iloc[i-1,1]
        
            print("So after differencing the data one time, the data is :")
            print (data)
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            print("CHECKING WHETHER THE STATIONARITY IS REMOVED FROM THE DATA THROUGH VISUALIZATIONS AND COMPARING IT WITH THE PLOTS BEFORE DIFFERENCING THE DATA")
        
            #yt versus yt-1
            plt.scatter(data.iloc[1:-1,-1],data.iloc[2:,-1]) 
            plt.title('y_t versus y_t+1')
            plt.show()
            plt.close()
        
            #For a stationary process that the nature of the joint probability distribution 𝑝(𝑧 𝑡, 𝑧 𝑡+𝑘) of values separated by 𝑘 intervals of time can be inferred by plotting a scatter diagram using pairs of values(𝑧 𝑡, 𝑧 𝑡+𝑘) of the time series, separated by a constant interval or lag k
            lag_k = 730 # 2 years
        
            z_t = data.iloc[1:-lag_k,-1]  # Excluding the last 'k' values
            z_t_k = data.iloc[lag_k:-1,-1]  # Excluding the first 'k' values
        
            plt.figure(figsize=(8, 6))
            plt.scatter(z_t, z_t_k)
        
            plt.title(f"Scatter Plot of z_t vs z_t+{lag_k}")
            plt.xlabel("z_t")
            plt.ylabel(f"z_t+{lag_k}")
            plt.grid(True)
            plt.show()
            plt.close()
        
            #For different time lags and different sets of data
            plt.scatter(data.iloc[500:865,-1],data.iloc[500+365:865+365,-1])
            plt.title("Scatter Plot of y_500:865 vs y_865:1230")
            plt.xlabel("y_500:865")
            plt.ylabel("y_865:1230")
            plt.grid(True)
            plt.show()
            plt.close()
        
            plt.scatter(data.iloc[500:865,-1],data.iloc[500-365:865-365,-1])
            plt.title("Scatter Plot of y_500:865 vs y_135:500")
            plt.xlabel("y_500:865")
            plt.ylabel("y_135:500")
            plt.grid(True)
            plt.show()
            plt.close()
        
            print("There is a pattern followed in the data after differencing, which can be seen in all the scatter plots. The data is stationary now")
        
            #Checking for constant mean
            print("Mean of y_t  : ",data.iloc[1:-1,-1].mean())
            print("Mean of y_t+1: ",data.iloc[2:,-1].mean())
            print("The mean is constant for different time lags")
        
            #Plotting rolling mean and standard deviation
            rmean = data.iloc[1:,-1].rolling(window=30).mean().dropna()
            rstd = data.iloc[1:,-1].rolling(window=30).std().dropna()
            #print(rmean,rstd)
        
            plt.figure(figsize=(14,5))
        
            plt.plot(data['Evening'] , color='black',label='Original')
            plt.plot(rmean , color='red',label='Rolling Mean')
            plt.plot(rstd,color='blue',label = 'Rolling Standard Deviation')
        
            plt.title("Rolling mean and standard deviation")
            plt.legend(loc='best')
            plt.show()
            print("The Rolling mean is almost zero which is a sign of stationary data")
        
            #Time series plot
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Evening_Differenced_1'], color='blue', label='Differenced Evening Price')
            plt.title('Gold Differenced Evening Price Over Time')
            plt.xlabel('Date')
            plt.ylabel('Differenced Evening Price')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.close()
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
            plt.figure(figsize=(12, 6))
            plot_acf(data.iloc[1:,-1], lags=50)
            plt.title("Autocorrelation (ACF) for Differenced Evening Rate")
            plt.show()
        
            plt.figure(figsize=(12, 6))
            plot_pacf(data.iloc[1:,-1], lags=50)
            plt.title("Partial Autocorrelation (PACF) for Differenced Evening Rate")
            plt.show()
            print("From the above results we can infer that after differencing the data once, the data set is stationary")
        
            '''MODEL BUILDING AND MODEL EVALUATION AND FORECASTING'''
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            import numpy as np
        
            def ARIMA_MODEL():
                from statsmodels.tsa.arima.model import ARIMA
        
                train_series = pd.to_numeric(train_data['Evening_Differenced_1'], errors='coerce').dropna() # train_data.iloc[:,-1]
                train_series.index.freq = pd.infer_freq(train_series.index) # 'D
        
                test_series = pd.to_numeric(test_data['Evening_Differenced_1'], errors='coerce').dropna() # test_data.iloc[:,-1]
                test_series.index.freq = pd.infer_freq(test_series.index) # 'D
        
                model = ARIMA(train_series, order=(1,0,1)) # order = (1, 1, 1)
                arima_result = model.fit()
        
                forecast = arima_result.forecast(steps=len(test_series))
        
                #print(forecast)
                #print(test_series)
        
                # Model Evaluation
                mae = mean_absolute_error(test_series, forecast)
                mse = mean_squared_error(test_series, forecast)
                rmse = np.sqrt(mse)
                r2 = r2_score(test_series, forecast)
        
                print("ARIMA Model Evaluation :-")
                print("Mean Absolute Error (MAE):", mae)
                print("Mean Squared Error (MSE):", mse)
                print("Root Mean Squared Error (RMSE):", rmse)
                print("R-squared (R2 Score):", r2)
        
                original_series = train_data['Evening']
        
                forecast_diff = forecast
        
                last_original_value = original_series.iloc[-1]
        
                forecast_original = [last_original_value + forecast_diff.iloc[0]]
        
                for i in range(1, len(forecast_diff)):
                    forecast_original.append(forecast_original[-1] + forecast_diff.iloc[i])
        
                forecast_original = np.array(forecast_original)
                
                #print(forecast_original)
        
                # Convert back to a pandas Series for convenience
                forecast_original_series = pd.Series(forecast_original, index=test_series.index)
                #One thing to be noted here is that earlier the yhat values and another 12 or so features also came with the forecast which is the result of arima_result.forecast(steps=len(test_series))
        
                # Plot Actual vs Predicted
                plt.figure(figsize=(12, 6))
        
                plt.plot(train_series.index, train_series, label="Training Data", color='blue')
                plt.plot(test_series.index, test_series, label="Actual Test Data", color='green')
                plt.plot(test_series.index, forecast, label="Predicted Test Data", color='red')
        
                plt.title("Actual vs Predicted - ARIMA Model")
                plt.xlabel("Date")
                plt.ylabel("Evening_Differenced_1")
                plt.legend()
                plt.show()
        
                # Plot the original vs. reverted forecast
                plt.figure(figsize=(12, 6))
                plt.plot(test_series.index, test_series, label="Actual Data", color="blue")
                plt.plot(test_series.index, forecast_original_series, label="Reverted Forecast", color="red")
                plt.title("Actual vs. Forecast (Original Scale)")
                plt.xlabel("Date")
                plt.ylabel("Gold Price")
                plt.legend()
                plt.show()
        
            def LSTM_MODEL():
                from tensorflow.keras.models import Sequential # Defines a linear stack of layers for the LSTM model.
                from tensorflow.keras.layers import LSTM, Dense # Adds LSTM layers for capturing temporal dependencies in the data.
                                                            # Fully connected layer to output a single value (forecast).
                from sklearn.preprocessing import MinMaxScaler # Scales data between a specified range (0 to 1 here), which is essential for LSTM as it works better with scaled data.
        
                scaler = MinMaxScaler(feature_range=(0, 1)) # Class - MinMaxScaler
                                                        # Object - scaler
                                                        # It is a preprocessing technique to scale the data into a specified range which is 0 to 1
                                                        # x_new = a + ( (b-a) * ( x_old - min(x) ) / ( max(x) - min(x) ) )
        
        
                train_series = pd.to_numeric(train_data['Evening_Differenced_1'], errors='coerce').dropna() # train_data.iloc[:,-1]
                train_series.index.freq = pd.infer_freq(train_series.index) # 'D
        
                test_series = pd.to_numeric(test_data['Evening_Differenced_1'], errors='coerce').dropna() # test_data.iloc[:,-1]
                test_series.index.freq = pd.infer_freq(test_series.index) # 'D
        
                train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
                test_scaled = scaler.transform(test_series.values.reshape(-1, 1))
        
                # Function to create sequences for LSTM
                def create_sequences(data, seq_length):
                    x, y = [], []
                    for i in range(len(data) - seq_length):
                        x.append(data[i:i + seq_length])
                        y.append(data[i + seq_length])
                    return np.array(x), np.array(y)
        
                seq_length = 10  # Number of time steps in each sequence
                x_train, y_train = create_sequences(train_scaled, seq_length)
                x_test, y_test = create_sequences(test_scaled, seq_length)
        
                # Reshape input for LSTM (samples, time steps, features)
                # x_train.shape[0] - ( len(train_data) - seq_length ) - number of 2d-matrices or samples
                # x_train.shape[1] - seq_length or time steps         - number of data in a sequence = 10
                # third is the feature which is 1
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
                # Build LSTM model
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                    LSTM(50, activation='relu'),
                    Dense(1)
                ])
        
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)
        
                # Predict on test data
                predicted_scaled = model.predict(x_test)
                predicted = scaler.inverse_transform(predicted_scaled)
        
                # Actual test data (corresponding y_test values need to be scaled back) # Actually instead of calculating wecan use the test_data from above right?
                actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
                # Model Evaluation
                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)
        
                print("LSTM Model Evaluation :-")
                print("Mean Absolute Error (MAE):", mae)
                print("Mean Squared Error (MSE):", mse)
                print("Root Mean Squared Error (RMSE):", rmse)
                print("R-squared (R2 Score):", r2)
        
                # Plot Actual vs Predicted
                plt.figure(figsize=(14,10))
        
                plt.plot(test_series.index[seq_length:], actual, label="Actual Test Data", color='green')
                plt.plot(test_series.index[seq_length:], predicted, label="Predicted Test Data", color='red')
        
                plt.title("Actual vs Predicted - LSTM Model")
                plt.xlabel("Date")
                plt.ylabel("Evening_Differenced_1")
                plt.legend()
                plt.show()
        
            def PROPHET_MODEL():
                from prophet import Prophet
                train_series = pd.to_numeric(train_data['Evening_Differenced_1'], errors='coerce').dropna() # train_data.iloc[:,-1]
                train_series.index.freq = pd.infer_freq(train_series.index) # 'D
        
                test_series = pd.to_numeric(test_data['Evening_Differenced_1'], errors='coerce').dropna() # test_data.iloc[:,-1]
                test_series.index.freq = pd.infer_freq(test_series.index) # 'D
        
                # Prepare the data for Prophet
                train_df = train_series.reset_index()
                test_df = test_series.reset_index()
        
                # Prophet requires two columns: 'ds' for date and 'y' for the values
                train_df.columns = ['ds', 'y']
                test_df.columns = ['ds', 'y']
        
                # Initialize and fit the Prophet model
                model = Prophet()
                model.fit(train_df)
        
                # Forecast for the test data period
                future = pd.DataFrame({'ds': test_df['ds']})
                forecast = model.predict(future)
        
                # Extract the forecasted values
                forecasted_values = forecast['yhat'].values
        
                # Model Evaluation
                mae = mean_absolute_error(test_df['y'], forecasted_values)
                mse = mean_squared_error(test_df['y'], forecasted_values)
                rmse = np.sqrt(mse)
                r2 = r2_score(test_df['y'], forecasted_values)
        
                print("Prophet Model Evaluation :-")
                print("Mean Absolute Error (MAE):", mae)
                print("Mean Squared Error (MSE):", mse)
                print("Root Mean Squared Error (RMSE):", rmse)
                print("R-squared (R2 Score):", r2)
        
                # Plot Actual vs Predicted
                plt.figure(figsize=(12, 6))
        
                plt.plot(test_df['ds'], test_df['y'], label="Actual Test Data", color='green')
                plt.plot(test_df['ds'], forecasted_values, label="Predicted Test Data", color='red')
        
                plt.title("Actual vs Predicted - Prophet Model")
                plt.xlabel("Date")
                plt.ylabel("Evening_Differenced_1")
                plt.legend()
                plt.show()
        
                # Prophet's built-in plot
                #model.plot(forecast)
                #plt.title("Prophet Forecast with Components")
                #plt.show()
        
                # Get the last known value from the training dataset
                last_original_value = train_data['Evening'].iloc[-1]
        
                # Reconstruct the forecasted original values
                reconstructed_forecast = [last_original_value]  # Initialize with the last original value
                for diff_value in forecasted_values:
                    next_value = reconstructed_forecast[-1] + diff_value
                    reconstructed_forecast.append(next_value)
        
                # Remove the initial value as it's only for reconstruction
                reconstructed_forecast = reconstructed_forecast[1:]
        
                # Convert to a pandas DataFrame for easier plotting
                reconstructed_df = pd.DataFrame({
                    'ds': test_df['ds'],
                    'actual': test_data['Evening'].values,  # Original test data
                    'predicted': reconstructed_forecast
                })
        
                # Plot Actual vs Reconstructed
                plt.figure(figsize=(12, 6))
                plt.plot(reconstructed_df['ds'], reconstructed_df['actual'], label="Actual Data", color='green')
                plt.plot(reconstructed_df['ds'], reconstructed_df['predicted'], label="Reconstructed Predicted Data", color='red')
                plt.title("Actual vs Predicted - Reconstructed")
                plt.xlabel("Date")
                plt.ylabel("Evening Gold Prices")
                plt.legend()
                plt.show()
        
            train_data = data.iloc[1:-30]
            test_data = data.iloc[-30:]
        
            ARIMA_MODEL()
            LSTM_MODEL()
            PROPHET_MODEL()
            print("The Prophet model works better compared to other models.")
            st.session_state.is_analysis_done = True
        
if __name__ == "__main__":
    main()
