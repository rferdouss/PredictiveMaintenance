# FeatureExtraction-For-SupervisedML -
# A python script to extract specific features for formulating multivariant time series forecasting problem into
# supervised ML problem. This is done to use different ML and DL solutions
#
#
# Raihana Ferdous
# Date :  19/06/2024


import numpy as np
import os
import pickle
from collections import OrderedDict
import random
import math
import datetime as dt
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit



#Global variables
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"

outputfeaturelist = trainComp+"-Feature"
intervaltime = 1 #Sampling the data   # weekly =7 , bi-weekly =15, monthly =30
thresholdPercentageMaxAlertInADay = 0.50   # value range = [0 -1], 0.70 means 70% of the max alert and number of alert is considered

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
    alertabovemaxalert=0
    alertabovemaxcriticalalert=0

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
        if(datecount<=intervaltime):
            nummaintenance= nummaintenance+ (TrainDataDiagosticMaintenance[datekey]['Maintenance'])
            numalert = numalert + len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])
            count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[datekey]['AlertLevel']) if i == 1]
            alertcode.append(TrainDataDiagosticMaintenance[datekey]['CriticalAlertCode'])
            alertcolor.append(TrainDataDiagosticMaintenance[datekey]['CriticalAlertColor'])
            numcriticalalert =numcriticalalert + len(count_occ_critical)

            if(len(TrainDataDiagosticMaintenance[datekey]['Alertkey']) >= MaxAlertThreshold): # check if the alert for this day exceeds of the threshold of max alert in  a day
                numdaywithmaxalert = numdaywithmaxalert+1
                alertabovemaxalert=1

            if( len(count_occ_critical) >=  MaxCriticalAlertThreshold):
                numdaywithmaxcriticalalert = numdaywithmaxcriticalalert + 1
                alertabovemaxcriticalalert

        if(datecount>=intervaltime):
            #datekey.month, '  Year = ', datekey.year
            enddate = datekey
            if (numalert>0 and intervaltime>0):
                avgalert = numalert / intervaltime

            if(numcriticalalert>0 and intervaltime>0):
                avgcriticalalert = numcriticalalert/intervaltime

            #print('alert code = ', alertcode)
            #print('alert color = ', alertcolor)
            TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month':datekey.month, 'Year': datekey.year, 'StartDate':startdate, 'EndDate':enddate,'Maintenance': nummaintenance, 'NumAlerts': numalert, 'NumCriticalAlerts':numcriticalalert, 'CriticalAlertCodeList':alertcode, 'CriticalAlertColor': alertcolor, 'AvgAlert': avgalert, 'AvgCriticalAlert': avgcriticalalert, 'AlertAboveThreshold': alertabovemaxalert, 'CriticalAlertAboveThreshold':alertabovemaxcriticalalert,'NumDayWithMaxAlert': numdaywithmaxalert, 'NumDayWithMaxCriticalAlert': numdaywithmaxcriticalalert}
            index=index+1
            datecount = 0
            numalert = 0
            nummaintenance = 0
            numcriticalalert = 0
            alertcode = []
            alertcolor = []
            alertabovemaxalert = 0
            alertabovemaxcriticalalert = 0


        datecount=datecount+1
    print('Size of the bin after sampling it with ',  intervaltime, '(days) interval = ', len(TrainMaintenanceDiagnosticInfoForAnTimeInterval.keys()))
    #print(TrainMaintenanceDiagnosticInfoForAnTimeInterval)
    #store the features in a pickle file
    outputdataAnalysisfile = outputlocation + '\\' + trainID + '-' + outputfeaturelist
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))

    return TrainMaintenanceDiagnosticInfoForAnTimeInterval
#-----------------------------------------------------------------------------------------------------------------
# Feature extraction
#------------------------------------------------------------------------------------------------------------------
def FeatureExtraction(subdfTrainData):
    #label_col = ['alert', 'calert']
    data = pd.DataFrame(subdfTrainData[['NumAlerts', 'NumCriticalAlerts', 'AlertAboveThreshold']].copy())
    #data.columns = [label_col]

    # add the lag of the target variable from 6 steps back up to 24
    for i in range(7, 9):
        data['Alertlag_{}'.format(i)] = data['NumAlerts'].shift(i)

    for i in range(7, 9):
        data['Criticallag_{}'.format(i)] = data['NumCriticalAlerts'].shift(i)

    lag_cols_alerr = [col for col in data.columns if 'Alertlag' in col]
    data['rolling_mean_Alert'] = data[lag_cols_alerr].mean(axis=1)

    lag_cols = [col for col in data.columns if 'Criticallag' in col]
    data['rolling_mean_Critical'] = data[lag_cols].mean(axis=1)


    # extract out the features and labels into separate variables
    #y = data[label_col].values
    data = data.drop('NumAlerts', axis=1)
    data = data.drop('NumCriticalAlerts', axis=1)

    X = data.values
    feature_names = data.columns
    print('dimension: ', X.shape)
    data.head()

    print('Feature extraction = ', data.tail(10))

    return data
#----------------------------------------------------------------------------------------------------------
#Function : timeseries_train_test_split(X, y, test_size)
#----------------------------------------------------------------------------------------------------------
def timeseries_train_test_split(X, y, test_size):
    """Perform train-test split with respect to time series structure."""
    test_index = int(len(X) * (1 - test_size))
    X_train = X[:test_index]
    X_test = X[test_index:]
    y_train = y[:test_index]
    y_test = y[test_index:]
    return X_train, X_test, y_train, y_test

#----------------------------------------------------------------------------------------------------------
#Function : mean_absolute_percentage_error(y_true, y_pred)
#----------------------------------------------------------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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
    print('Number of da ys in diagnostic file = ', len(TrainDataDiagosticMaintenance.keys()), '  Bin/sampling size (days)= ', intervaltime)
    #print(TrainDataRecord)
    #DrawAlertCodeColorForSpecificInterval(TrainDataRecord)

    #-------------Processing of daily Time Series Data into a sample of week/bi-week/monthly slots and extracts features------------------------------------
    dfTrainData = pd.DataFrame.from_dict(TrainDataRecord, orient='index')
    #subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts','Maintenance']]  # Create new pandas DataFrame
    subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts', 'Maintenance','AvgAlert','AvgCriticalAlert', 'AlertAboveThreshold','CriticalAlertAboveThreshold','NumDayWithMaxAlert', 'NumDayWithMaxCriticalAlert']]  # Create new pandas DataFrame

    # ----------------------------------------------------------------------------------------------------------------
    # Feature extraction : convert time series into a format applicable for using Supervised Machine Learning techniques
    # Given this 3 dimensional time series data, we will need to do some feature-engineering to make it applicable for a
    # downstream supervised learning method. Including:
    # 1. Generating lagged features and window statistics from them.
    # 2. We will also label the output - if the alerts exceeds a threshold (boolean feature indicating whether the val exceeds a threshold).
    # ----------------------------------------------------------------------------------------------------------------
    dataf = FeatureExtraction(subdfTrainData)
    traindatadividepercentage = 0.70  # 75% of the data will be used for training, 25% data will be used for testing
    size = int(len(dataf) * traindatadividepercentage)
    train_df, test_df = dataf[0:size], dataf[size:len(dataf)]
    print('training data size  = ', len(train_df))
    #trainy = train_df.iloc[:, 0]
    trainX, trainy = train_df.iloc[:,1:], train_df.iloc[:, 0]
    testX, testy = test_df.iloc[:, 1:], test_df.iloc[:, 0]
    #print('train y = ', trainX)

    # we are using random forest here, feel free to swap this out
    # with your favorite regression model
    model = RandomForestRegressor(max_depth=6, n_estimators=50)
    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    print('prediction = ', prediction)

    plt.figure(figsize=(15, 7))

    x = range(prediction.size)
    plt.plot(x, prediction, label='prediction', linewidth=2.0)
    plt.plot(x, testy, label='actual', linewidth=2.0)
    error = mean_absolute_percentage_error(prediction, testy)
    plt.title('Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()





    #for i in range(len(test)):
# split test row into input and output columns


    #X_train = X[:test_index]
    #X_test = X[test_index:]
    #y_train = y[:test_index]
    #y_test = y[test_index:]

    #UnivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
    #MultivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
