import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import tqdm as tqdm

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def get_data():
    train_df= pd.read_csv("train.csv")
    data = train_df[['first_day_of_month', 'microbusiness_density']].copy(deep = True)
    data['Date'] = pd.to_datetime(data['first_day_of_month'])
    data = data.drop(columns = 'first_day_of_month')
    data = data.set_index('Date')
    data = data.groupby(data.index).mean()
    return data

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    '''
    Code for plotting the time series plot
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

def visual_inspection():
    data = get_data()
    dec = sm.tsa.seasonal_decompose(data['microbusiness_density'],period = 12,model = 'additive').plot()
    plt.show()

def data_diffs():
    data = get_data()
    data_diff = data.diff()
    data_diff = data_diff.dropna()

    dec = sm.tsa.seasonal_decompose(data_diff,period = 12).plot()
    plt.show()
    return data_diff

def seasonal_diff():
    data_diff = data_diffs()
    data_diff_seas = data_diff.diff(12)
    data_diff_seas = data_diff_seas.dropna()
    dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 12)
    dec.plot()
    plt.show()
