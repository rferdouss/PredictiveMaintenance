# FeatureExtraction-For-SupervisedML -
# A python script to extract specific features for formulating multivariant time series forecasting problem into
# supervised ML problem. This is done to use different ML and DL solutions
#
#
# Raihana Ferdous
# Date :  17/06/2024


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
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



#Global variables
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"


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

#----------------------------------------------------------------------------------------
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)): # find the end of this pattern
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
            # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)




#------------------------------------------------------------------------------------
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    #print(TrainDataRecord)
    #DrawAlertCodeColorForSpecificInterval(TrainDataRecord)

    #---------------------------------------Preparing different monthly plots (visualization)----------------------------
    #outputfilelocation = outputlocation + '\\TSR'+trainnumber+'-TCU-DiagnosticData'
    #print('Monthly critical alert coMonltyBoxPlotAlertCriticalSingleTrainde and color distribution for single train------------------------')
    #MonltyBoxPlotAlertCriticalSingleTrain(outputfilelocation)  # this one is correct
    #MonltyPlotCriticalAlertCodeColorSingleTrain(outputfilelocation)
    # --------------------------------------------------------------------------------------------------------

    #-------------Time Series Data processing for building various forecasting models------------------------------------
    dfTrainData = pd.DataFrame.from_dict(TrainDataRecord, orient='index')
    subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts']]  # Create new pandas DataFrame
    varname = 'NumAlerts'  #'NumAlerts', 'NumCriticalAlerts'
    #print(subdfTrainData.head(10))

    #-------------Univariate time series forecasting using LSTM models------------------------------------

    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # define model
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    n_features = 1
    n_seq = 2
    n_steps = 2

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # demonstrate prediction
    x_input = np.array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print('Prediction val = ', yhat)