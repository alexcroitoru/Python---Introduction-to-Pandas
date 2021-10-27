#!/usr/bin/env python
# coding: utf-8

#  #  A Whale off the Port(folio)
#  ---
# 
#  In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500 Index.

# In[ ]:


# Initial imports
import pandas as pd
import numpy as np
from datetime import datetime as dt
from pathlib import Path
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Cleaning
# 
# In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.
# 
# Files:
# 
# * `whale_returns.csv`: Contains returns of some famous "whale" investors' portfolios.
# 
# * `algo_returns.csv`: Contains returns from the in-house trading algorithms from Harold's company.
# 
# * `sp500_history.csv`: Contains historical closing prices of the S&P 500 Index.

# ## Whale Returns
# 
# Read the Whale Portfolio daily returns and clean the data

# In[ ]:


# Reading whale returns
path=os.getcwd()
csv_path1=path+"\\Resources\whale_returns.csv"
whale_returns = pd.read_csv(csv_path1, index_col="Date")
whale_returns.sort_values("Date")


# In[ ]:


# Count nulls
whale_returns.isnull().sum()


# In[ ]:


# Drop nulls
whale_returns = whale_returns.dropna().copy()


# ## Algorithmic Daily Returns
# 
# Read the algorithmic daily returns and clean the data

# In[ ]:


# Reading algorithmic returns
path=os.getcwd()
csv_path2=path+"\\Resources\algo_returns.csv"
algo_returns = pd.read_csv(csv_path2, index_col="Date")
algo_returns.sort_values("Date")


# In[ ]:


# Count nulls
algo_returns.isnull().sum()


# In[ ]:


# Drop nulls
algo_returns = algo_returns.dropna().copy()


# ## S&P 500 Returns
# 
# Read the S&P 500 historic closing prices and create a new daily returns DataFrame from the data. 

# In[ ]:


# Reading S&P 500 Closing Prices
path=os.getcwd()
csv_path3=path+"\\Resources\sp500_history.csv"
sp500_history = pd.read_csv(csv_path3)


# In[ ]:


# Check Data Types
sp500_history.dtypes


# In[ ]:


# Fix Data Types
sp500_history["Close"] = sp500_history["Close"].str.replace("$", "")
sp500_history["Close"]
sp500_history["Close"] = sp500_history["Close"].astype("float")
sp500_history['Date'] = pd.to_datetime(sp500_history['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')
sp500_history.set_index(sp500_history['Date'], inplace=True)
sp500_history = sp500_history.drop(columns=["Date"])
sp500_history.sort_values("Date")
sp500_history = sp500_history.iloc[::-1]
sp500_history.head()


# In[ ]:


# Calculate Daily Returns
sp500_daily_returns = sp500_history.pct_change()


# In[ ]:


# Drop nulls
sp500_daily_returns.isnull().sum()


# In[ ]:


sp500_daily_returns = sp500_daily_returns.dropna().copy()


# In[ ]:


# Rename `Close` Column to be specific to this portfolio.
sp500_returns = sp500_daily_returns.rename(columns={
    "Close": "SP500 Daily Return",})
sp500_df = sp500_returns


# ## Combine Whale, Algorithmic, and S&P 500 Returns

# In[ ]:


# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.
join_returns = pd.merge(algo_returns, whale_returns, on='Date', how='inner')
joined_returns = pd.merge(sp500_df, join_returns, on='Date', how='inner')


# ---

# # Conduct Quantitative Analysis
# 
# In this section, you will calculate and visualize performance and risk metrics for the portfolios.

# ## Performance Anlysis
# 
# #### Calculate and Plot the daily returns.

# In[ ]:


# Plot daily returns of all portfolios
returns_plots = joined_returns.plot()
returns_plots = joined_returns.plot(subplots=True)
fig = plt.figure()


# #### Calculate and Plot cumulative returns.

# In[ ]:


# Calculate cumulative returns of all portfolios
cumulative_ret = (joined_returns + 1).cumprod()


# In[ ]:


# Plot cumulative returns
cumulative_plots = cumulative_ret.plot(subplots=True)
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(cumulative_ret)
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("Portfolio Cumulative Returns")
plt.show();


# ---

# ## Risk Analysis
# 
# Determine the _risk_ of each portfolio:
# 
# 1. Create a box plot for each portfolio. 
# 2. Calculate the standard deviation for all portfolios
# 4. Determine which portfolios are riskier than the S&P 500
# 5. Calculate the Annualized Standard Deviation

# ### Create a box plot for each portfolio
# 

# In[ ]:


# Box plot to visually show risk
returns_std = joined_returns.std()
sp500_std_plots = sp500_df.plot.box()
whales_std_plots = whale_returns.plot.box()
algos_std_plots = algo_returns.plot.box()


# ### Calculate Standard Deviations

# In[ ]:


# Calculate the daily standard deviations of all portfolios
returns_std


# ### Determine which portfolios are riskier than the S&P 500

# In[ ]:


# Calculate  the daily standard deviation of S&P 500
daily_std = returns_std.sort_values(ascending=False)
# Determine which portfolios are riskier than the S&P 500
print(daily_std)
print(f'_________')
print(f'These portfolios are riskier than the S&P 500:')
print(f"BERKSHIRE HATHAWAY INC")
print(f'TIGER GLOBAL MANAGEMENT LLC')


# ### Calculate the Annualized Standard Deviation

# In[ ]:


# Calculate the annualized standard deviation (252 trading days)
annualized_std = returns_std * np.sqrt(252)
annualized_std.head(7)


# ---

# ## Rolling Statistics
# 
# Risk changes over time. Analyze the rolling statistics for Risk and Beta. 
# 
# 1. Calculate and plot the rolling standard deviation for the S&P 500 using a 21-day window
# 2. Calculate the correlation between each stock to determine which portfolios may mimick the S&P 500
# 3. Choose one portfolio, then calculate and plot the 60-day rolling beta between it and the S&P 500

# ### Calculate and plot rolling `std` for all portfolios with 21-day window

# In[ ]:


# Calculate the rolling standard deviation for all portfolios using a 21-day window
joined_returns.rolling(window=21).std()


# In[ ]:


# Plot the rolling standard deviation
joined_returns.rolling(window=21).std().plot()


# ### Calculate and plot the correlation

# In[ ]:


# Calculate the correlation
price_correlation = joined_returns.corr()


# In[ ]:


# Display de correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(price_correlation.corr(), fignum=f.number)
plt.xticks(range(price_correlation.select_dtypes(['number']).shape[1]), price_correlation.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(price_correlation.select_dtypes(['number']).shape[1]), price_correlation.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# ### Calculate and Plot Beta for a chosen portfolio and the S&P 500

# In[ ]:


# Calculate covariance of a single portfolio
covariance = joined_returns["SOROS FUND MANAGEMENT LLC"].cov(joined_returns["SP500 Daily Return"])


# In[ ]:


# Calculate variance of S&P 500
variance = joined_returns['SP500 Daily Return'].var()


# In[ ]:


# Computing beta
soros_beta = covariance / variance
soros_beta


# In[ ]:


# Plot beta trend
sns.lmplot(x='SP500 Daily Return', y='SOROS FUND MANAGEMENT LLC', data=joined_returns, aspect=2.5, fit_reg=True)


# ## Rolling Statistics Challenge: Exponentially Weighted Average 
# 
# An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the [`ewm`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html) with a 21-day half-life.

# In[ ]:


# Use `ewm` to calculate the rolling window
returns_ewm = joined_returns.ewm(halflife=21).mean()
returns_ewm


# ---

# # Sharpe Ratios
# In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. After all, if you could invest in one of two portfolios, and each offered the same 10% return, yet one offered lower risk, you'd take that one, right?
# 
# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[ ]:


# Annualized Sharpe Ratios
sharpe_ratio = ((joined_returns.mean()-joined_returns['SP500 Daily Return'].mean()) * 252) / (joined_returns.std() * np.sqrt(252))
sharpe_ratio


# In[ ]:


# Visualize the sharpe ratios as a bar plot
sharpe_ratio.plot(kind="bar", title="Sharpe Ratios")


# ### Determine whether the algorithmic strategies outperform both the market (S&P 500) and the whales portfolios.
# 
# Write your answer here!
# 
# Based on the Sharpe Ratio, Standard Deviation, and Rate of Return - Algo 1 presents itself as outperforming all the portfolios evaluated and to a lesser degree, Algo 2 accomplishes the same. 

# ---

# # Create Custom Portfolio
# 
# In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 
# 
# 1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
# 3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns
# 4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others
# 5. Include correlation analysis to determine which stocks (if any) are correlated

# ## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 
# For this demo solution, we fetch data from three companies listes in the S&P 500 index.
# 
# * `GOOG` - [Google, LLC](https://en.wikipedia.org/wiki/Google)
# 
# * `AAPL` - [Apple Inc.](https://en.wikipedia.org/wiki/Apple_Inc.)
# 
# * `COST` - [Costco Wholesale Corporation](https://en.wikipedia.org/wiki/Costco)

# In[ ]:


# Reading data from 1st stock
path=os.getcwd()
csv_path10=path+"\\Resources\goog_historical.csv"
goog_return = pd.read_csv(csv_path10)
goog_return['Trade DATE'] = pd.to_datetime(goog_return['Trade DATE'], dayfirst=True).dt.strftime('%Y-%m-%d')
goog_returns = goog_return.sort_values("Trade DATE")
goog_returns = goog_returns.rename(columns={
    "Trade DATE" : "Date",
    "NOCP": "GOOG Price",
})
goog_returns.set_index(goog_returns['Date'], inplace=True)
goog_returns = goog_returns.drop(columns=["Date", "Symbol"])


# In[ ]:


# Reading data from 2nd stock
path=os.getcwd()
csv_path11=path+"\\Resources/aapl_historical.csv"
aapl_return = pd.read_csv(csv_path11)
aapl_return['Trade DATE'] = pd.to_datetime(aapl_return['Trade DATE'], dayfirst=True).dt.strftime('%Y-%m-%d')
aapl_returns = aapl_return.sort_values("Trade DATE")
aapl_returns = aapl_returns.rename(columns={
    "Trade DATE" : "Date",
    "NOCP": "AAPL Price",
})
aapl_returns.set_index(aapl_returns['Date'], inplace=True)
aapl_returns = aapl_returns.drop(columns=["Date", "Symbol"])


# In[ ]:


# Reading data from 3rd stock
path=os.getcwd()
csv_path12=path+"\\Resources/cost_historical.csv"
cost_return = pd.read_csv(csv_path12)
cost_return['Trade DATE'] = pd.to_datetime(cost_return['Trade DATE'], dayfirst=True).dt.strftime('%Y-%m-%d')
cost_returns = cost_return.sort_values("Trade DATE")
cost_returns = cost_returns.rename(columns={
    "Trade DATE": "Date",
    "NOCP": "COST Price",
})
cost_returns.set_index(cost_returns['Date'], inplace=True)
cost_returns = cost_returns.drop(columns=["Date","Symbol"])


# In[ ]:


# Combine all stocks in a single DataFrame
stocks_returns = pd.concat([cost_returns, goog_returns, aapl_returns], axis="columns", join="inner")


# In[ ]:


# Calculate daily returns
stocks_dailyreturns = stocks_returns.pct_change()
stocks_dailyreturns.head()
# Drop NAs
stocks_dailyreturns = stocks_dailyreturns.dropna().copy()
# Display sample data
stocks_dailyreturns.tail()


# ## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

# In[ ]:


# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
weighted_portfolio = stocks_dailyreturns.dot(weights)
weighted_portfolio.sum

# Display sample data
weighted_portfolio


# ## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

# In[ ]:


# Join your returns DataFrame to the original returns DataFrame
all_returns = pd.concat([joined_returns, weighted_portfolio], axis="columns", join="inner")
columns = ['SP500 Daily Return', 'Algo 1', 'Algo 2', 'SOROS FUND MANAGEMENT LLC', 'PAULSON & CO.INC.', 'TIGER GLOBAL MANAGEMENT LLC', 'BERKSHIRE HATHAWAY INC', 'Portfolio']
all_returns.columns = columns


# In[ ]:


# Only compare dates where return data exists for all the stocks (drop NaNs)
all_returns


# In[ ]:


cumulative_return = (all_returns + 1).cumprod()


# ## Re-run the risk analysis with your portfolio to see how it compares to the others

# ### Calculate the Annualized Standard Deviation

# In[ ]:


# Calculate the annualized `std`
returns_std_total = all_returns.std()
annualized_std_total = returns_std_total * np.sqrt(252)
annualized_std_total.head(8)


# ### Calculate and plot rolling `std` with 21-day window

# In[ ]:


# Calculate rolling standard deviation
all_returns.rolling(window=21).std()
# Plot rolling standard deviation
all_returns.rolling(window=21).std().plot()


# ### Calculate and plot the correlation

# In[ ]:


# Calculate and plot the correlation
portfolio_correlation = all_returns.corr()
# Display de correlation matrix
portfolio_correlation


# In[ ]:


g = plt.figure(figsize=(19, 15))
plt.matshow(portfolio_correlation.corr(), fignum=g.number)
plt.xticks(range(portfolio_correlation.select_dtypes(['number']).shape[1]), portfolio_correlation.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(portfolio_correlation.select_dtypes(['number']).shape[1]), portfolio_correlation.select_dtypes(['number']).columns, fontsize=14)
cd = plt.colorbar()
cd.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# ### Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500

# In[ ]:


# Calculate and plot Beta
covariance_port = all_returns["Portfolio"].rolling(60).cov(all_returns["SP500 Daily Return"])
variance_port = all_returns['SP500 Daily Return'].rolling(60).var()
portfolio_beta = covariance_port / variance_port
portfolio_beta.plot()


# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[ ]:


# Calculate Annualzied Sharpe Ratios
sharpe_ratio_all = ((all_returns.mean()-all_returns['SP500 Daily Return'].mean()) * 252) / (all_returns.std() * np.sqrt(252))


# In[ ]:


# Visualize the sharpe ratios as a bar plot
sharpe_ratio_all.plot(kind="bar", title="Sharpe Ratios")


# ### How does your portfolio do?
# 
# Write your answer here!
# 
# Our Portfolio - comprised of only three stocks, assumes a much higher standard deviation than an index portfolio, that being said, it seems that the higher the risk, the higher reward in this case. The Sharpe Ratio.
# Our portfolio generates higher returns than all but the Algo 1 Portfolio, which is a very solid result. 

# In[ ]:


cumulative_plot = cumulative_return.plot(subplots=True)
figure = plt.figure()
ax2 = figure.add_axes([0.1,0.1,0.8,0.8])
ax2.plot(cumulative_return)
ax2.set_xlabel('Date')
ax2.set_ylabel("Cumulative Returns")
ax2.set_title("Portfolio Cumulative Returns")
plt.show();


# In[ ]:




