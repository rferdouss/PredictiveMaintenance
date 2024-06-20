# Multivariate multistep time series forecasting using LSTM -
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
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

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D



#Global variables
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"

outputfeaturelist = trainComp+"-Feature"
intervaltime = 7  #Sampling the data   # weekly =7 , bi-weekly =15, monthly =30
thresholdPercentageMaxAlertInADay = 0.70   # value range = [0 -1], 0.70 means 70% of the max alert and number of alert is considered

def GetMaxAlert(TrainData):
    maxalert = 0
    maxcriticalalert =0
    print('function  - get max alert')
    count_occ_critical=[]
    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    for datekey in TrainDataDiagosticMaintenance.keys():
        print('date =  ', datekey, '   num alert = ', len(TrainDataDiagosticMaintenance[datekey]['Alertkey']), '  Critical = ', len(count_occ_critical))
        if (len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])>maxalert):
            maxalert = len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])

        count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[datekey]['AlertLevel']) if i == 1]
        if (len(count_occ_critical)>maxcriticalalert):
            maxcriticalalert = len(count_occ_critical)

        #print('date =  ', datekey, '   num alert = ', len(TrainDataDiagosticMaintenance[datekey]['Alertkey']),
        #      '  Critical = ', len(count_occ_critical), '    Maxalet  =', maxalert, '   Max critical = ', maxcriticalalert)
    return maxalert, maxcriticalalert

#---------------------------------------------------------------------------------------------------------------------------
#Function : Alert count for an interval
def GetAlertforADuration(TrainData, trainID, intervaltime, maxalert, maxcriticalalert):
    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    #print('Function GetAlertforADuration() - Get alert information for a time interval (days)= ', intervaltime)
    TrainMaintenanceDiagnosticInfoForAnTimeInterval={}

    datecount=0
    numalert=0
    nummaintenance=0
    numcriticalalert=0
    avgalert=0
    avgcriticalalert=0
    numdaywithmaxalert=0
    numdaywithmaxcriticalalert=0

    alertcode=[]
    alertcolor=[]
    index=1
    startdate =0
    enddate=0

    MaxAlertThreshold = maxalert * thresholdPercentageMaxAlertInADay
    MaxCriticalAlertThreshold = maxcriticalalert * thresholdPercentageMaxAlertInADay

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

            if(len(TrainDataDiagosticMaintenance[datekey]['Alertkey']) > MaxAlertThreshold): # check if the alert for this day exceeds of the threshold of max alert in  a day
                numdaywithmaxalert = numdaywithmaxalert+1

            if( len(count_occ_critical) >  MaxCriticalAlertThreshold):
                numdaywithmaxcriticalalert = numdaywithmaxcriticalalert + 1

        if(datecount>=intervaltime):
            #datekey.month, '  Year = ', datekey.year
            enddate = datekey
            if (numalert>0 and intervaltime>0):
                avgalert = numalert / intervaltime

            if(numcriticalalert>0 and intervaltime>0):
                avgcriticalalert = numcriticalalert/intervaltime

            #print('alert code = ', alertcode)
            #print('alert color = ', alertcolor)
            TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month':datekey.month, 'Year': datekey.year, 'StartDate':startdate, 'EndDate':enddate,'Maintenance': nummaintenance, 'NumAlerts': numalert, 'NumCriticalAlerts':numcriticalalert, 'CriticalAlertCodeList':alertcode, 'CriticalAlertColor': alertcolor, 'AvgAlert': avgalert, 'AvgCriticalAlert': avgcriticalalert, 'NumDayWithMaxAlert': numdaywithmaxalert, 'NumDayWithMaxCriticalAlert': numdaywithmaxcriticalalert}
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
    #store the features in a pickle file
    outputdataAnalysisfile = outputlocation + '\\' + trainID + '-' + outputfeaturelist
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))

    return TrainMaintenanceDiagnosticInfoForAnTimeInterval

#----------------------------------------------------------------------------------------------------------
#Function : LoadProcessedTrainData(filename) - method to load the processed data stored in a pickle file
def LoadProcessedTrainData(filename):
    TrainDataD={}
    TrainDataD = pickle.load(open(filename, "rb"))
    #print(TrainDataD)
    return TrainDataD
#----------------------------------------------------------------------------------------------------------------
# Function DataScaling() - All the columns in the data frame are on a different scale. Now we will scale the
# values to -1 to 1 for faster training of the models
#---------------------------------------------------------------------------------------------------------------
def DataScaling(train_df, test_df):
    train = train_df
    scalers = {}
    for i in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        train[i] = s_s
    test = test_df
    for i in train_df.columns:
        scaler = scalers['scaler_' + i]
        s_s = scaler.transform(test[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        test[i] = s_s
    return train, test

#---------------------------------------------------------------------------------------------------
#Function ConvertTimeSeriesToSupervisedLearningSamples(series, n_past, n_future) - Converting the series to samples
#use a sliding window approach to transform our series into samples of input past observations and output future observations to use supervised learning algorithms
#Input:  n_past ==> no of past observations, n_future ==> no of future observations
#----------------------------------------------------------------------------------------------------
def ConvertTimeSeriesToSupervisedLearningSamples(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)
#------------------------------------------------------------------------------------
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainname ='TSR'
    trainnumber = '070'  # T1 =070, T2 = 086, T3 =040
    traincomponent ='TCU'
    trainDiagnosticfilename =dataInputLocation +"\\DiagnosticData\\"+trainname+" "+trainnumber+".csv"
    trainMaintenanceDatafilename = dataInputLocation +"\\MaintenanceData\\Avvisi SAP - Interventi manutentivi.xlsx"
    datasheetname =trainname+" "+trainnumber
    trainID = trainname+trainnumber


#----------------------------------------------------------------------------------------------------------------
# Load processed data that are stored in pickle files
#----------------------------------------------------------------------------------------------------------------
    print('Loading processed data file for prediction analysis : ')
    outputpicklefilename= outputlocation + '\\'+ trainID + '-'+outputfname # diagnostic and maintenance data
    print(outputpicklefilename)
    TrainDataDiagosticMaintenance = LoadProcessedTrainData(outputpicklefilename)
    #print(TrainDataDiagosticMaintenance)

# ----------------------------------------------------------------------------------------------------------------
# Analysis the data
# ----------------------------------------------------------------------------------------------------------------
    maxnumalertperday = 0
    maxnumcriticalalertperday = 0
    maxcriticalalertcategoryperday = 0
    maxnumalertperday, maxnumcriticalalertperday = GetMaxAlert(TrainDataDiagosticMaintenance)
    print('For this train, Max alert in a day =  ', maxnumalertperday, '   Max Critical Alert in a day =  ', maxcriticalalertcategoryperday)

    # sample data according to chosen slot
    TrainDataRecord = GetAlertforADuration(TrainDataDiagosticMaintenance, trainID, intervaltime,maxnumalertperday, maxnumcriticalalertperday)
    print('Number of days in diagnostic file = ', len(TrainDataDiagosticMaintenance.keys()), '  Bin/sampling size (days)= ', intervaltime)
    #print(TrainDataRecord)
    #DrawAlertCodeColorForSpecificInterval(TrainDataRecord)

    #-------------Processing of daily Time Series Data into a sample of week/bi-week/monthly slots and extracts features------------------------------------
    dfTrainData = pd.DataFrame.from_dict(TrainDataRecord, orient='index')
    #subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts','Maintenance']]  # Create new pandas DataFrame
    #subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts', 'Maintenance','AvgAlert','AvgCriticalAlert', 'NumDayWithMaxAlert', 'NumDayWithMaxCriticalAlert']]  # Create new pandas DataFrame
    # For formulation 1:
    subdfTrainData = dfTrainData[['AvgAlert','AvgCriticalAlert', 'NumDayWithMaxAlert', 'NumDayWithMaxCriticalAlert']]  # Create new pandas DataFrame


    #print('data = ', subdfTrainData)
    #varname = 'NumAlerts'  #'NumAlerts', 'NumCriticalAlerts'

    #UnivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
    #MultivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)

    #prepare the dataset for the LSTM. This involves framing the dataset as a supervised learning problem and normalizing the input variables.
    #-----------------Step 1: Divide data into train and test set------------------------------------------------------------------------------
    traindatadividepercentage =0.75 # 75% of the data will be used for training, 25% data will be used for testing
    size = int(len(subdfTrainData) * traindatadividepercentage)
    train_df, test_df = subdfTrainData[0:size], subdfTrainData[size:len(subdfTrainData)]
    print('Divide data')
    # -----------------Step 2: Scaling the data---------------------------------------------------------------------------------------
    #train, test = DataScaling(train_df, test_df)
    #print('Step 1', test)
    train = train_df
    scalers = {}
    for i in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        train[i] = s_s
    test = test_df
    for i in train_df.columns:
        scaler = scalers['scaler_' + i]
        s_s = scaler.transform(test[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        test[i] = s_s

    #-----------------Step 3: Converting the series to samples -------------------------------------------------------------------------
    #for this case, let’s assume that given the past 10 weeks observation, we need to forecast the next 5 weeks observations.
    #n_past ==> no of past observations, n_future ==> no of future observations, n_features == > no of features at each timestep in the data.
    n_past = 15
    n_future = 5
    n_features = 4

    # Convert both training and testing set time series into supervised learning input format
    X_train, y_train = ConvertTimeSeriesToSupervisedLearningSamples(train.values, n_past, n_future) # training
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

    X_test, y_test = ConvertTimeSeriesToSupervisedLearningSamples(test.values, n_past, n_future) # testing
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

    #-----------------Step 4: Model Architecture ----------------------------------------------------------------------------------------
    #E1D1 ==> Sequence to Sequence Model with one encoder layer and one decoder layer.
    #E2D2 == > Sequence to Sequence Model with two encoder layers and two decoder layers.
    #------------------------------------------------------------------------------------------------------------------------------------
    # E1D1 ==> Sequence to Sequence Model with one encoder layer and one decoder layer.
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
    decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

    model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)
    print(model_e1d1.summary())

    #E2D2 == > Sequence to Sequence Model with two encoder layers and two decoder layers.
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])

    decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    print(model_e2d2.summary())

    #-------------------Step 5 : Training the model---------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------
    print('Training the model')
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    history_e1d1 = model_e1d1.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32,verbose=0, callbacks=[reduce_lr])

    #model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    #history_e2d2 = model_e2d2.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32,verbose=0, callbacks=[reduce_lr])

    pred_e1d1 = model_e1d1.predict(X_test)
    pred_e2d2 = model_e2d2.predict(X_test)
    #print(pred_e1d1)
    print('before rescaling =', y_test[1, :, 1], pred_e2d2[1, :, 1])
    #plt.figure(figsize=(16, 9))
    #plt.plot(history_e1d1.history['loss'])
    #plt.plot(history_e1d1.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train loss', 'validation loss'])
    #plt.show()



    #scalers = {}
    #print('columns = ', train_df.columns)
    #-----Inverse Scaling of the predicted values
    for index, i in enumerate(train_df.columns):
        #print('index = ',pred_e1d1[:, :, index])
        scaler = scalers['scaler_'+i]
        pred_e1d1[:, :, index] = scaler.inverse_transform(pred_e1d1[:, :, index])
        pred_e1d1[:, :, index] = scaler.inverse_transform(pred_e1d1[:, :, index])
        pred_e2d2[:, :, index] = scaler.inverse_transform(pred_e2d2[:, :, index])
        pred_e2d2[:, :, index] = scaler.inverse_transform(pred_e2d2[:, :, index])
        y_train[:, :, index] = scaler.inverse_transform(y_train[:, :, index])
        y_test[:, :, index] = scaler.inverse_transform(y_test[:, :, index])

    for index, i in enumerate(train_df.columns):
        print(i)
        for j in range(1, 6):
            print("Day ", j, ":")
            print("MAE-E1D1 : ", mean_absolute_error(y_test[:, j - 1, index], pred_e1d1[:, j - 1, index]), end=", ")
            print("MAE-E2D2 : ", mean_absolute_error(y_test[:, j - 1, index], pred_e2d2[:, j - 1, index]))
        print()
        print()

    print('plot', y_test[1, : , 1],pred_e2d2[1, :, 1] )

    plt.figure(figsize=(16, 9))
    plt.plot(list(y_test[1, : , 1]))
    plt.plot(list(pred_e2d2[1, :, 1]))
    plt.title("Actual vs Predicted")
    plt.ylabel("Avg Alert")
    plt.legend(('Actual', 'predicted'))
    plt.show()
