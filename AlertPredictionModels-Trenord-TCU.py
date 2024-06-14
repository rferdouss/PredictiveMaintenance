# MLPredictiveMaintenance -
# A python script to test different statistical models and find a suitable one for predicting number of alerts or critical alerts for the TCU unit
#Univariate time-series forecasting methods
# Raihana Ferdous
# Date :  04/04/2024


import csv
from collections import defaultdict
import numpy as np
import os
import pickle
from collections import OrderedDict
from datetime import datetime
import random
import math
import datetime as dt
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as mpl
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
#import scipy.stats as stats
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA

import pmdarima as pm
from pmdarima.arima import auto_arima
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing        # for Holt-Winters Exponential Smoothing


#Global variables
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"



#---------------------------------------------------------------------------------------------
#Function: CheckStationarityInTimeSeriesData()
def CheckStationarityInTimeSeriesData(subdfTrainData, varname, rolling_window):
    print('-----------------------------------------------------------------')
    print('Check for stationarity ---------')
    # Determing rolling statistics
    rolmean =subdfTrainData[varname].rolling(rolling_window).mean() # pd.rolling_mean(subdfTrainData[varname], window=12)
    rolstd = subdfTrainData[varname].rolling(rolling_window).std() #pd.rolling_std(subdfTrainData[varname], window=12)
    #print(rolmean)

    # Plot rolling statistics:
    plt.plot(subdfTrainData[varname], color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    print('----------------------Results of Dickey-Fuller Test:--------------------')
    dftest = adfuller(subdfTrainData[varname], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)



#--------------------------------------------------------------------------------------------
# Function: Prepare data for time series analysis (prepare pandas dataframe from dictionary)
def DecompositionOfTimeSeriesData(subdfTrainData, varname, tinterval):
    print('-----------------------------------------------------------------------')
    print('Time series decomposition into trend and seasonality ')
    # Additive Decomposition
    additive_decomposition = seasonal_decompose(subdfTrainData[varname], model='additive', period=tinterval, extrapolate_trend='freq')

    # Plot
    #plt.rcParams.update({'figure.figsize': (16, 12)})
    #multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.xlabel('Number of weeks')
    #plt.ylabel('Number of critical alarms')
    #plt.title('')
    plt.show()

    # Extract the Decomposition Components ----
    # Actual Values = Product of (Seasonal * Trend * Resid)
    df_reconstructed = pd.concat([additive_decomposition.seasonal, additive_decomposition.trend, additive_decomposition.resid, additive_decomposition.observed], axis=1)
    df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
    print(df_reconstructed.head())


#-----------------------------------------------------------------------------------------
#Function : RollingStatistics(subdfTrainData, varname) [for checking trend in time series data]
def RollingStatistics(subdfTrainData, varname):
    print('-----------------------------------------------------------------------')
    print('Determine rolling statistics for data')
    windowsize = subdfTrainData[varname].size
    print('window size: ', windowsize)
    subdfTrainData["rolling_avg"] = subdfTrainData[varname].rolling(window=windowsize).mean()  # window size 12 denotes 12 months, giving rolling mean at yearly level
    subdfTrainData["rolling_std"] = subdfTrainData[varname].rolling(window=windowsize).std()

    # Plot rolling statistics
    plt.figure(figsize=(15, 7))
    plt.plot(subdfTrainData[varname], color='#379BDB', label='Original')
    plt.plot(subdfTrainData["rolling_avg"], color='#D22A0D', label='Rolling Mean')
    plt.plot(subdfTrainData["rolling_std"], color='#142039', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
#-------------------------------------------------------------------------------------
#Function  DrawACFandPACFPlot(TrainDataRecord, varname)
#important to determing value of p and d before using ARIMA model
def DrawACFandPACFPlot(subdfTrainData, varname):
    lag_acf = acf(subdfTrainData[varname], nlags=12) # with lag 20 [meaning consider previous 40 data info]
    lag_pacf = pacf(subdfTrainData[varname], nlags=12, method='ols')
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(subdfTrainData[varname])), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(subdfTrainData[varname])), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(subdfTrainData[varname])), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(subdfTrainData[varname])), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    plt.show()

    print('ACF and PACF plots with 20 lags')
    plot_acf(subdfTrainData[varname], lags=20)
    plot_pacf(subdfTrainData[varname], lags=20)
    plt.show()
#=======================DIFFERENT FORECASTING MODELS=================================================
#Function : Naive forecasting model :
def NaiveForecastingModel(subdfTrainData, varname, traindatapercentage):
    print('BaseLine Model - Naive Forecasting Model')
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]
    #print('Train data = ', train[varname])

    #print('last index = ',len(train[varname]) - 1, '  Last val = ', train[varname][len(train[varname]) - 1])
    y_hat_naive = test.copy()
    y_hat_naive['naive_forecast'] = train[varname][len(train[varname]) - 1]
    print('Naive forecast =', y_hat_naive['naive_forecast'])

    # random prediction
    print('Random walk')
    actualtest =[]
    predicted=[]

    predictions = list()
    history = train[varname][len(train[varname])]
    print('history = ', history, 'test = ', len(train[varname]))

    for i in range(len(subdfTrainData[varname])+1):
        if i > (len(train[varname])):
            randomval = random.random()
            print('i val = ', i, 'random val = ', randomval)
            if randomval<0.5:
                yhat = history -1
            else:
                yhat = history +1
        #yhat = history + (-1 if np.random() < 0.5 else 1)

            print('test val , ', history, 'pred = ', yhat)
            predicted.append(yhat)
            actualtest.append(test[varname][i])
            #history = yhat
    print('out')
    print(actualtest, predicted)
    error = np.square(np.subtract(actualtest,predicted)).mean()#mean_squared_error(test[varname], dfp)
    rsmev = math.sqrt(error)
    print('Random WALK RMSE: %.3f' % rsmev)

    ytext = ''

    if (varname == 'NumAlerts'):
        ytext = 'Number of Alerts'

    if (varname == 'NumCriticalAlerts'): #'NumAlerts', 'NumCriticalAlerts'
        ytext = 'Number of Critical Alerts'

    #print('var name = ', varname, ytext)

    plt.figure(figsize=(20, 5))
    plt.grid()
    plt.plot(subdfTrainData[varname], label='Actual',linestyle='solid', linewidth=2.5, color='black')
    #plt.plot(test[varname], label='Test')
    plt.plot(y_hat_naive['naive_forecast'], label='Predicted', linestyle='--', linewidth=4, color='red')
    plt.legend(loc='best')
    plt.title('Random Walk Model', fontsize=20, fontweight='bold')
    plt.xlabel('Number of weeks', fontsize=20, fontweight='bold')
    plt.ylabel(ytext, fontsize=20, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend(prop={"size": 20, "weight": 'bold'})
    plt.show()

    rmse = np.sqrt(mean_squared_error(test[varname], y_hat_naive['naive_forecast'])).round(2)
    mape = mean_absolute_percentage_error(test[varname], y_hat_naive['naive_forecast']) * 100

    results = pd.DataFrame({'Method': ['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    print(results)
    return rmse, mape


#---------------------------------------------------------------------------------------
#Function: Time series forecasting and prediction using Statistical Model - MA (Moving Average)
#-------------------------------------------------------------------------------------------------------------
def MovingAverage_Model(subdfTrainData, varname, traindatapercentage, windowSize):
    ytext = ''

    if (varname == 'NumAlerts'):
        ytext = 'Number of Alerts'

    if (varname == 'NumCriticalAlerts'):  # 'NumAlerts', 'NumCriticalAlerts'
        ytext = 'Number of Critical Alerts'


    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]
    window =windowSize

    y_hat_avg = test.copy()
    y_hat_avg['moving_avg_forecast'] = train[varname].rolling(window).mean().iloc[-1]   #window=window_size
    plt.figure(figsize=(20, 6))
    plt.plot(subdfTrainData[varname], label='Actual', linestyle='solid', linewidth=2.5, color='black')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Predicted', linestyle='--', linewidth=4, color='red')

    plt.xlabel('Number of weeks', fontsize=20, fontweight='bold')
    plt.ylabel(ytext, fontsize=20, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend(prop={"size": 20, "weight": 'bold'})
    plt.legend(loc='best')
    plt.title('Moving Average Model', fontsize=20, fontweight='bold')
    plt.show()

    rmse = np.sqrt(mean_squared_error(test[varname], y_hat_avg['moving_avg_forecast'])).round(2)
    mape = mean_absolute_percentage_error(test[varname], y_hat_avg['moving_avg_forecast']) * 100

    results = pd.DataFrame({'Method': ['Moving Average method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    print(results)

    return rmse, mape
#---------------------------------------------------------------------------------------
#Function : Double Exponential Smoothing (for series with trend)
def DoubleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage):
    print('--------------------------Exponential Smoothing ----------------------------------')
    print('Double exponential smoothing')
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]
    # Double exponential smoothing (for time series with only trend)
    des_model = ExponentialSmoothing(train[varname], trend="add").fit(smoothing_level=0.5,smoothing_trend=0.5)
    #des_model = ExponentialSmoothing(train[varname], trend="add").fit(smoothing_level=0.5, smoothing_trend=0.5) # correct

    forecast_period = len(test[varname])  # number of next forecast
    # Forecast next 5 periods
    y_pred = des_model.forecast(forecast_period)

    rms = sqrt(mean_squared_error(test[varname], y_pred))
    print('rms = ', rms) # rooot mean square error

    ytext = ''

    if (varname == 'NumAlerts'):
        ytext = 'Number of Alerts'

    if (varname == 'NumCriticalAlerts'):  # 'NumAlerts', 'NumCriticalAlerts'
        ytext = 'Number of Critical Alerts'

    # y_pred = tes_model.forecast(48)
    plt.figure(figsize=(20, 6))
    plt.plot(subdfTrainData[varname], label='Actual', linestyle='solid', linewidth=2.5, color='black')
    plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=4, color='red')

    plt.xlabel('Number of weeks', fontsize=20, fontweight='bold')
    plt.ylabel(ytext, fontsize=20, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend(prop={"size": 20, "weight": 'bold'})
    plt.legend()
    plt.title('Exponential Smoothing (simple) Model', fontsize=20, fontweight='bold')
    plt.show()

    return rms
#---------------------------------------------------------------------------------------
#Function : Triple Exponential Smoothing (Holt-Winters)
def TripleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage):
    print('--------------------------Exponential Smoothing ----------------------------------')
    print('Triple exponential smoothing')
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]
    # Double exponential smoothing (for time series with only trend)
    # des_model = ExponentialSmoothing(train[varname], trend="add").fit(smoothing_level=0.5,smoothing_trend=0.5)

    forecast_period = len(test[varname])  # number of next forecast
    # y_pred = des_model.forecast(forecast_period)

    # Triple Exponential Smoothing -  Fit data
    #model = ExponentialSmoothing(
    #    train[varname],
    #    seasonal_periods=4,
    #    trend="add",
    #    seasonal="add",
    #    use_boxcox=True,
    #    initialization_method="estimated",
    #).fit()

    # Triple exponential smoothing(Holt-Winters)(for time series with both trend and seasonality)
    tes_model = ExponentialSmoothing(train[varname],trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5,
                                                              smoothing_slope=0.5, smoothing_seasonal=0.5)

    #tes_model = ExponentialSmoothing(subdfTrainData[varname],
    #                                 trend="add", seasonal="add",seasonal_periods=12).fit(smoothing_level=0.5,smoothing_slope=0.5,smoothing_seasonal=0.5)

    # Forecast next 5 periods
    y_pred = tes_model.forecast(forecast_period)

    rms = sqrt(mean_squared_error(test[varname], y_pred))
    print('rms = ', rms) # rooot mean square error

    ytext = ''

    if (varname == 'NumAlerts'):
        ytext = 'Number of Alerts'

    if (varname == 'NumCriticalAlerts'):  # 'NumAlerts', 'NumCriticalAlerts'
        ytext = 'Number of Critical Alerts'
    # y_pred = tes_model.forecast(48)
    plt.figure(figsize=(20, 6))
    plt.plot(subdfTrainData[varname], label='Actual', linestyle='solid', linewidth=2.5, color='black')
    plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=4, color='red')

    plt.xlabel('Number of weeks', fontsize=20, fontweight='bold')
    plt.ylabel(ytext, fontsize=20, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend(prop={"size": 20, "weight": 'bold'})
    plt.legend()
    plt.title('Triple Exponential Smoothing Model', fontsize=20, fontweight='bold')
    plt.show()

    return rms

#---------------------------------------------------------------------------------------
#Function: Time series forecasting and prediction using Statistical Model - AR
#-------------------------------------------------------------------------------------------------------------
def ARMA_Model(subdfTrainData, varname, traindatapercentage):
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]

    model = AutoReg(train[varname], lags=10)
    #model = AutoReg(train[varname], order=(0, 1))

    model_fit = model.fit()
    forecast = model_fit.predict(len(test[varname]))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
    print('forecast = ', forecast)

    #Plotting forecast
    plt.plot(train[varname], color="black")
    plt.plot(test[varname], color="red")
    plt.plot(forecast, color="blue")
    plt.title('ARMA - forecasting')
    plt.ylabel(varname)
    plt.xlabel('Time (Number of week)')
    plt.show()

    rmse = np.sqrt(mean_squared_error(test[varname], forecast)).round(2)
    mape = mean_absolute_percentage_error(test[varname], forecast) * 100

    results = pd.DataFrame({'Method': ['ARMA method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    print(results)

    return rmse, mape

#---------------------------------------------------------------------------------------
#Function: Time series forecasting and prediction using Statistical Model - ARIMA
#-------------------------------------------------------------------------------------------------------------
def ARIMA_Model(subdfTrainData, varname, traindatapercentage):
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]

    model = auto_arima(train[varname],  start_p=0, start_q=0, test='adf', m=12, seasonal=False, trace=True)

    model.fit(train[varname])
    forecast = model.predict(n_periods=len(test[varname]))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

    #Plotting forecast
    plt.plot(train[varname], color="black")
    plt.plot(test[varname], color="red")
    plt.plot(forecast, color="blue")
    plt.title('ARIMA - forecasting')
    plt.ylabel(varname)
    plt.xlabel('Time (Number of week)')
    plt.show()

    #print(test[varname])
    #forecast['Prediction'][116]=101.011629
    #forecast.loc[116, 'Prediction'] = 140.011629
    print('forecast', forecast)

    rmse = np.sqrt(mean_squared_error(test[varname], forecast)).round(2)
    mape = mean_absolute_percentage_error(test[varname], forecast) * 100

    results = pd.DataFrame({'Method': ['ARIMA method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    print(results)

    return rmse, mape

#------------------------------------------------------------------------------------------
#Function : SARIMA_Model()
def SARIMA_Model(subdfTrainData, varname, traindatapercentage):
    print('-------------------SARIMA Model-----------------------------------------')
    size = int(len(subdfTrainData) * traindatapercentage)
    train, test = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]

    # finding the best model with the training data
    model = auto_arima(train[varname], start_p=1, start_q=1, test='adf', m=12, seasonal=False, trace=True)
    print(model)
    # fit the model with the test data


    #sarima = ARIMA(train[varname], order=(0, 1, 1))
    sarima = SARIMAX(train[varname], order=(0, 1, 1), seasonal_order=(0, 0, 0, 12))
    #print('SARIMA training length = ', len(train[varname]))
    predicted = sarima.fit().predict(start=(len(train[varname])-1), end=(len(subdfTrainData[varname])+1))

    #print('Actual = ', test[varname])
    predicted.drop(predicted.tail(1).index, inplace=True)  # drop last n rows
    predicted.drop(predicted.head(2).index, inplace=True)  # drop last n rows
    #print('SARIMA Predicted = ', predicted)

    #model = sarima.fit()
    #predicted = model.predict(n_periods=len(test[varname]))
    #print(predicted)
    # Get confidence intervals of forecasts
    #pred_ci = predicted.conf_int()

    ytext = ''

    if (varname == 'NumAlerts'):
        ytext = 'Number of Alerts'

    if (varname == 'NumCriticalAlerts'):  # 'NumAlerts', 'NumCriticalAlerts'
        ytext = 'Number of Critical Alerts'

    plt.figure(figsize=(20, 6))
    plt.plot(subdfTrainData[varname], label='Actual',linestyle='solid', linewidth=2.5, color='black')
    plt.plot(predicted, label='Predicted', linestyle='--', linewidth=4, color='red')
    #plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.xlabel('Number of weeks', fontsize=20, fontweight='bold')
    plt.ylabel(ytext, fontsize=20, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend(prop={"size": 20, "weight": 'bold'})
    plt.title('SARIMA Model', fontsize=20, fontweight='bold')
    plt.show()

    resid = test[varname] - predicted
    mae = abs(resid.mean())
    print(mae)  #mean absolute error

    rmse = np.sqrt(mean_squared_error(test[varname], predicted)).round(2)
    mape = mean_absolute_percentage_error(test[varname], predicted) * 100


    results = pd.DataFrame({'Method': ['SARIMA'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    print(results)

    return rmse, mape

#-------------------------------Lagging plot -----------------------------------------
def LaggingPlot(subdfTrainData, varname):
    # Lagging Plot  - to check the correlation between time and specific feature (here number of alerts/ critical alerts)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(subdfTrainData[varname], lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))


    fig.suptitle('Lag Plots of Air Passengers', y=1.05)
    plt.show()

#--------------Evaluation metric :  Mean Absolute Scaled Error (MASE)------------------------
def MASE(Actual, Predicted):
    values = []
    for i in range(1, len(Actual)):
        values.append(abs(Actual[i] - Predicted[i]) / (abs(Actual[i] - Actual[i - 1]) / len(Actual) - 1))
    return np.mean(values)

#---------------------------------------------------------------------------------------------------------------------------
#Function : Alert count for an interval
def GetAlertforADuration(TrainData, trainID, intervaltime):
    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    #print('Function GetAlertforADuration() - Get alert information for a time interval (days)= ', intervaltime)
    TrainMaintenanceDiagnosticInfoForAnTimeInterval={}

    datecount=0
    numalert=0
    nummaintenance=0
    numcriticalalert=0
    alertcode=[]
    alertcolor=[]
    index=1
    startdate =0
    enddate=0
    for datekey in TrainDataDiagosticMaintenance.keys():
        if(datecount ==0):
            startdate =  datekey
            enddate = 0
        if(datecount<intervaltime):
            nummaintenance= nummaintenance+ (TrainDataDiagosticMaintenance[datekey]['Maintenance'])
            numalert = numalert + len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])
            count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[datekey]['AlertLevel']) if i == 1]
            alertcode.append(TrainDataDiagosticMaintenance[datekey]['CriticalAlertCode'])
            alertcolor.append(TrainDataDiagosticMaintenance[datekey]['CriticalAlertColor'])
            numcriticalalert =numcriticalalert + len(count_occ_critical)

        if(datecount>=intervaltime):
            #datekey.month, '  Year = ', datekey.year
            enddate = datekey
            #print('alert code = ', alertcode)
            #print('alert color = ', alertcolor)
            TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month':datekey.month, 'Year': datekey.year, 'StartDate':startdate, 'EndDate':enddate,'Maintenance': nummaintenance, 'NumAlerts': numalert, 'NumCriticalAlerts':numcriticalalert, 'CriticalAlertCodeList':alertcode, 'CriticalAlertColor': alertcolor}
            index=index+1
            datecount = 0
            numalert = 0
            nummaintenance = 0
            numcriticalalert = 0
            alertcode = []
            alertcolor = []


        datecount=datecount+1
    print('Size of the bin after sampling it with ',  intervaltime, '(days) interval = ', len(TrainMaintenanceDiagnosticInfoForAnTimeInterval.keys()))
    #print(TrainMaintenanceDiagnosticInfoForAnTimeInterval)
    return TrainMaintenanceDiagnosticInfoForAnTimeInterval

#----------------------------------------------------------------------------------------------------------
#Function : LoadProcessedTrainData(filename) - method to load the processed data stored in a pickle file
def LoadProcessedTrainData(filename):
    TrainDataD={}
    TrainDataD = pickle.load(open(filename, "rb"))
    #print(TrainDataD)
    return TrainDataD



#------------------------------------------------------------------------------------
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #processdataflag= False # True = perform data processing, False= load processed data and draw plots

    trainname ='TSR'
    trainnumber = '033'  # T1 =070, T2 = 086, T3 =040
    traincomponent ='TCU'
    trainDiagnosticfilename =dataInputLocation +"\\DiagnosticData\\"+trainname+" "+trainnumber+".csv"
    trainMaintenanceDatafilename = dataInputLocation +"\\MaintenanceData\\Avvisi SAP - Interventi manutentivi.xlsx"
    datasheetname =trainname+" "+trainnumber
    trainID = trainname+trainnumber


#----------------------------------------------------------------------------------------------------------------
# Load processed data that are stored in pickle files
#----------------------------------------------------------------------------------------------------------------
    print('Loading processed data file for prediction analysis : ')
    outputpicklefilename= outputlocation + '\\'+ trainID + '-'+outputdiagnosticfname # only diagnostic data
    print(outputpicklefilename)
    TrainDataDiagosticMaintenance = LoadProcessedTrainData(outputpicklefilename)
    #print(TrainDataDiagosticMaintenance)

# ----------------------------------------------------------------------------------------------------------------
# Analysis the data
# ----------------------------------------------------------------------------------------------------------------
    intervaltime = 7 # 15 days # Sampling the data
    TrainDataRecord = GetAlertforADuration(TrainDataDiagosticMaintenance, trainID, intervaltime)
    print('Number of days in diagnostic file = ', len(TrainDataDiagosticMaintenance.keys()), '  Bin/sampling size (days)= ', intervaltime)


    #-------------Time Series Data processing for building various forecasting models------------------------------------
    dfTrainData = pd.DataFrame.from_dict(TrainDataRecord, orient='index')
    subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts']]  # Create new pandas DataFrame
    varname = 'NumAlerts'  #'NumAlerts', 'NumCriticalAlerts'
    #print(subdfTrainData.head(10))

    # -------------Various Time Series Data processing for understanding underlying pattern for making decisions in forecasting----------------------------------
    autocorrelation_plot(subdfTrainData[varname]) # #AutocorrelationCheck()  # check for autocorrelation
    LaggingPlot(subdfTrainData,varname)
    autocorrelation_lag3 = subdfTrainData[varname].autocorr(lag=3)
    autocorrelation_lag5 = subdfTrainData[varname].autocorr(lag=5)
    autocorrelation_lag10 = subdfTrainData[varname].autocorr(lag=10)
    #autocorrelation_lag15 = subdfTrainData[varname].autocorr(lag=15)
    #autocorrelation_lag20 = subdfTrainData[varname].autocorr(lag=20)
    #print("Three Month Lag: ", autocorrelation_lag3,autocorrelation_lag5,autocorrelation_lag10,autocorrelation_lag15,autocorrelation_lag20)


    DecompositionOfTimeSeriesData(subdfTrainData, varname, intervaltime)  # decomposition of time series into trend and seasonality
    #RollingStatistics(subdfTrainData, varname)
    DrawACFandPACFPlot(subdfTrainData, varname)
    CheckStationarityInTimeSeriesData(subdfTrainData, varname, 12)  # check for stationary in data

    # ----------Experiments with different Time series forecasting models------------------------------------------------------------------------
    traindatapercentage= 0.80
    NaiveRMSE = NaiveForecastingModel(subdfTrainData, varname, traindatapercentage)

    windowsize = 12  # here 10 weeks
    MA_RMSE = MovingAverage_Model(subdfTrainData, varname, traindatapercentage, windowsize)
    arima_RMSE=0
    #arima_RMSE = ARIMA_Model(subdfTrainData, varname, traindatapercentage)#(subdfTrainData, varname, traindatapercentage)
    sarima_RMSE = SARIMA_Model(subdfTrainData, varname, traindatapercentage)
    tripleexponentialSmoothing_RMSE = TripleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage)
    doubleexponentialSmoothing_RMSE = DoubleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage)

    ARMA_RMSE=0
    #ARMA_RMSE = ARMA_Model(subdfTrainData, varname, traindatapercentage)

    print('Root Mean Square Error, ARIMA = ', arima_RMSE, '   SARIMA = ', sarima_RMSE)
    print('BaseLine Naive = ', NaiveRMSE)
    print('Moving Average (MA) = ', MA_RMSE)
    print('ARMA Model = ', ARMA_RMSE)
    print('ARIMA = ', arima_RMSE)
    print('SARIMA = ', sarima_RMSE)
    print('Double Exponential Smoothing = ', doubleexponentialSmoothing_RMSE)
    print('Triple Exponential Smoothing = ', tripleexponentialSmoothing_RMSE)

