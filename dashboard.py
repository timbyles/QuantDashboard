from numpy.core.fromnumeric import mean, shape, std
from numpy.linalg import norm
from pandas.io.formats.format import return_docstring
from scipy.stats.morestats import Mean
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import streamlit as st
import time
import scipy.stats as stats
import scipy.optimize as sc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import yfinance as yf
from yahoofinancials import YahooFinancials
import concurrent.futures as cf
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from statsmodels.tsa.stattools import coint
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing

st.title("Quant Dashboard")

st.sidebar.title("Options")

option = st.sidebar.selectbox("Which Dashboard?", ("Home", "Cointegration Search", "FOMO Friday", "Efficent Frontier", "Value Investing"))

if option == "Home":
    st.write("Please note that data is correct from  1/1/2018 to 1/1/2021. Up to date/extended date price data is too slow pulling from yfinance or without direct access to database.")
    st.write("This data comes from the current ASX300 and excludes any stocks that have a NULL price at any point in the above date frame. ")
    st.write("Currently only functional for stocks on the  ASX (Australia) exchange")
    st.write("Price data is Close data, not Adjusted Close. Will correct this before deployed properly, however Close data will suffice for proof of concept.")
    
if option == "Cointegration Search":
    
    st.header(option)

    st.write("The following code seeks to find stocks that are cointegrated to our chosen stock. This tells us that there is some relationship between the two stocks.")
    st.write("In the graphs generated below (if any) the ratios between the two stocks are shown and graphed in blue. The ratio is calculated with the chosen symbol as the numerator.")
    st.write("The solid line represents the average ratio spread between the two stocks over a given time period.")
    st.write("The dashed line represents +1 and -1 standard deviations away from the median, meaning that any spread within these two lines occurs 68% of the time (and 32% outside of them).")
    st.write("To determine cointegration an augmented Engle-Granger test is used, and only pairs that have a pvalue less than 0.05 are included.")

    data = pd.read_csv("ASX300 20182021.csv")
    data.set_index("Date", inplace=True, drop=True)

    country = st.sidebar.text_input("Country", "Australia").upper()
    symbol = st.sidebar.text_input("Symbol", "WOW").upper()

    countryDict = {"USA":"", "AUSTRALIA":".AX"}

    ticker = symbol + countryDict[country]

    pvalueList = []

    for stock in data:
        result = coint(data[ticker], data[stock])
        score = result[0]
        pvalue = result[1]
        if pvalue < 0.05 and stock != ticker:
            pvalueList.append([stock, pvalue])
    
    st.subheader("Cointegrated Stocks of " + symbol)

    stock1 = data[ticker]
    plt_index = 0

    st.write("The following stocks are cointegrated (according to the above criteria) with " + symbol + ":")
    st.write(" ")
    
    for items in pvalueList:
        stock2 = data[items[0]]
        ratio = stock1/stock2

        st.write(items[0])

        fig = px.line(ratio)
        fig.update_layout(xaxis_title="Date", yaxis_title="Ratio Spread", showlegend=False)
        fig.add_hline(y=ratio.mean())
        fig.add_hline(y=ratio.mean() + np.std(ratio), line_dash="dot")
        fig.add_hline(y=ratio.mean() - np.std(ratio), line_dash="dot")
        st.plotly_chart(fig)

if option == "FOMO Friday":

    st.header(option)

    st.write("A wishful legend in many stock forums is that FOMO Friday, being the fear of missing out over any news events on the weekend, is the phenomenon of price increases reserved for the trading action on Fridays,.")
    st.write("It is suggested that traders (and possibly institutional investors) will take the opportunity to form their positions in the event of new information coming during a closed market")
    st.write("Below is a linear regression model to determine if the price action (specifcally the percentage change) of Mon-Thurs, will have an effect on the close price on Friday.")

    urlSymbols = 'https://raw.githubusercontent.com/timbyles/ASX-Cointegration/main/ASX300.csv'
    df = pd.read_csv(urlSymbols)

    urlPriceData = 'https://raw.githubusercontent.com/timbyles/Quant-Dashboard/main/ASX300%2020182021.csv'
    data = pd.read_csv(urlPriceData)
    data.set_index("Date", inplace=True, drop=True)

    country = st.sidebar.text_input("Country", "Australia").upper()
    symbol = st.sidebar.text_input("Symbol", "WOW").upper()

    countryDict = {"USA":"", "AUSTRALIA":".AX"}

    ticker = symbol + countryDict[country]

    daysList = []

    for index, dates in enumerate(data.index):
        get_datetime = datetime.strptime(dates, '%Y-%m-%d')
        get_day = get_datetime.date().strftime("%A")
        close_price = data[ticker][index]
        daysList.append([get_datetime, get_day, close_price])

    # Inserts data into dataframe
    daysPercentChange = pd.DataFrame(daysList, columns=['Date','Day','Close Price'])

    # Gets Percentage Change of Close Price column and adds to dataframe. Drops top row as it will be NaN.
    percentChangeCol = daysPercentChange['Close Price'].pct_change()
    daysPercentChange["Percent Change"] = percentChangeCol
    daysPercentChange = daysPercentChange[1:]

    # Resets the week to begin at Monday.
    while daysPercentChange['Day'].iloc[0] != "Monday":
        daysPercentChange.drop(daysPercentChange.index[0], axis=0, inplace=True)

    # Cuts data off to finish at Friday so there are no half weeks.
    while daysPercentChange['Day'].iloc[-1] != "Friday":
        daysPercentChange.drop(daysPercentChange.tail(1).index,inplace=True) 
    
    daysPercentChange.reset_index(drop=True, inplace=True)

    # Finds amount of weeks to be tested, cuts off last week to prevent out of bounds indexing error.
    weeks = int(len(daysPercentChange)/5)-7

    weekPctChange = []

    # Loops through days and looks for valid weeks (Mon-Fri)
    for i in range(weeks):
        validWeekDf = daysPercentChange.head()
        #validWeekDf

        weekPctChange.append([validWeekDf['Percent Change'].iloc[0],
                         validWeekDf['Percent Change'].iloc[1],
                         validWeekDf['Percent Change'].iloc[2],
                         validWeekDf['Percent Change'].iloc[3],
                         validWeekDf['Percent Change'].iloc[4]])
       
        rules = [validWeekDf['Day'].iloc[0] == "Monday",
                 validWeekDf['Day'].iloc[1] == "Tuesday",
                 validWeekDf['Day'].iloc[2] == "Wednesday",
                 validWeekDf['Day'].iloc[3] == "Thursday",
                 validWeekDf['Day'].iloc[4] == "Friday"]   

        count = sum(rules)

        if count == 5:
            daysPercentChange = daysPercentChange.iloc[count:]
    
        check = False
        
        if daysPercentChange['Day'].iloc[0] == "Monday" and count !=5:
            daysPercentChange = daysPercentChange.iloc[4:]

        while check == False:
            if daysPercentChange['Day'].iloc[0] == "Monday":
                check = True
            else:
                daysPercentChange = daysPercentChange.iloc[1:]

    groupedWeeks = pd.DataFrame(weekPctChange, columns=['Mon','Tues','Wed', 'Thurs', 'Fri'])

    # Splits data into features and labels.
    X = groupedWeeks.iloc[:,:4]
    y = groupedWeeks.iloc[:,4:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    # Creates a data frame of test vs predictions.
    y_predict = pd.DataFrame(y_predict)
    y_test = y_test.reset_index(drop=True)
    df = pd.concat([y_predict, y_test], axis=1)
    df = df.set_axis(['Predict', 'Test'], axis=1, inplace=False)
    
    # Finds the results where predictions achieved correct direction.
    correct = 0

    for i in range(len(df.index)):
        P = df.iloc[i]['Predict']
        T = df.iloc[i]['Test']

        if (P < 0 and T < 0) or (P > 0 and T > 0) or (P == 0 and T == 0):
            correct += 1

    percentCorrect = round((correct/len(df.index))*100,2)
    
    st.subheader("Model Validity")

    st.write("The model predicted the correct direction " + str(percentCorrect) + "% of the time")

    # Model test metrics
    r2score = r2_score(y_test,y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)

    if r2score <= 0:
        st.write("However the coefficent of determination was less than 0 (" + str(round(r2score,2)) + ") and therefore not a valid model")
    if r2score > 0:
        st.write("The coefficent of determination was greater than 0 (" + str(round(r2score,2)) + ") and therefore can be considered a valid model, although as the coefficent appraoches 1, look out for overfitting")

    st.write("MAE = " + str(mae))
    st.write("MSE = " + str(mse))

    st.subheader("User Inputs")

    # User inputs to for testing model
    FriClose = st.number_input('Friday Close (previous week)', min_value=0.0, max_value=500.0, value=38.70, step=0.01)
    MonClose = st.number_input('Monday Close', min_value=0.0, max_value=500.0, value=38.87, step=0.01)
    TueClose = st.number_input('Tuesday Close', min_value=0.0, max_value=500.0, value=39.30, step=0.01)
    WedClose = st.number_input('Wednesday Close', min_value=0.0, max_value=500.0, value=39.11, step=0.01)
    ThursClose = st.number_input('Thursday Close', min_value=0.0, max_value=500.0, value=39.70, step=0.01)

    st.subheader("Predictions and Results")

    def pctChange(current,previous):
        pc = (current-previous)/previous
        return pc

    newTest = pd.DataFrame([[pctChange(MonClose,FriClose), pctChange(TueClose,MonClose), pctChange(WedClose,TueClose), pctChange(ThursClose,WedClose)]])

    change = round(model.predict(newTest)[0][0], 4)
    newPred = ThursClose*(1 + change)

    groupedWeeksList = groupedWeeks.to_numpy()
    flatNumpy = groupedWeeksList.flatten()  
    zscore = stats.zscore(flatNumpy)
    ave = np.mean(flatNumpy)
    stddev = np.std(flatNumpy)

    direction = "INCREASE"
    if change < 0:
        direction = "DECREASE" 

    z = (change - ave)/stddev

    st.write("The model predicts the Friday close price to be $" + str(round(newPred, 2)) + ", an " + direction +" of " + str(change*100) + "%")
    st.write("The z-score of this " + direction + " is " + str(round(z,2)))
    st.write("Presuming normal distrobution, this indicates a change that is " + str(round(stats.norm.cdf(z)*100,2)) + "% better than other changes within the dataset")
    st.write("For the dataset, the max z-score is " + str(round(np.max(zscore),2)) + " and the min z-score is " + str(round(np.min(zscore),2)))
 
    st.write("If I were to recode this, I would use log returns as opposed to normal returns. Log returns would be a more appropriate method as well as ensure price wont go below 0.")

if option == "Efficent Frontier":

    urlSymbols = 'https://raw.githubusercontent.com/timbyles/ASX-Cointegration/main/ASX300.csv'
    symbols = pd.read_csv(urlSymbols)

    symbolList = symbols['Code'].values.tolist()

    stockList = st.multiselect('ASX300 Stocks', symbolList, default=["A2M", "BHP", "CBA"])

    weights = np.empty(len(stockList))
    weights.fill(1/len(stockList))

    stocks = [stock + '.AX' for stock in stockList]

    startDate = '2018-01-01'
    endDate = datetime.today().strftime('%Y-%m-%d')
    
    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = pdr.DataReader(stock, data_source='yahoo', start=startDate, end=endDate)['Adj Close']

    returns = df.pct_change().dropna()
   
    # Annualized covariance matrix
    cov_matrix_annual = returns.cov() * 252

    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Calculate the annual portfolio return
    portfolio_simple_annual_return = np.sum(returns.mean()*weights)*252
 
    st.write("Given the choice of stocks, an equal weighted portfolio would have:")
    st.write("Annual Return: " + str(round(portfolio_simple_annual_return*100,2)) + "%")
    st.write("Volatility: " + str(round(portfolio_volatility*100,2)) + "%")

    # Calculate the expected returns and the annyalised sample covariance matrix of asset returns
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for max sharpe ratio. How much return in excess of risk free rate you receive per unit of volatility.
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    st.write(" ")
    st.write("After optimizing for Maximum Sharpe Ratio, the portfolio weights would become:")

    for i in cleaned_weights.keys():
        st.write("- " + i[0:3] + " : " + str(round(cleaned_weights[i]*100,2)) + "%")
   
    st.write("Annual Return: " + str(round(ef.portfolio_performance(verbose=True)[0]*100,2)) + "%")
    st.write("Annual Volatility: " + str(round(ef.portfolio_performance(verbose=True)[1]*100,2)) + "%")


