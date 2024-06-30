# TimeSeriesForecastingUsingMLAlgo -
# A python script to formulate a multi-variate time series forecasting problem into supervised machine learning problem
# Steps : Extract suitable features, apply various ML also, validate models
#
#
# Raihana Ferdous
# Date :  01/07/2024


import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
import random
import math
import datetime as dt
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import seaborn as sns


#Global variables
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"

outputfeaturelist = trainComp+"-Feature"

intervaltime = 7 #Sampling the data   # weekly =7 , bi-weekly =15, monthly =30
AlertThresholdPercentage = 0.50   # value range = [0 -1], 0.70 means 70% of the alert is considered

#---------------------------------------------------------------------------------------------------------------------
#Function - GetMaxAlert(TrainData)
#This function calculates the max number of alerts and critical alerts were observed in a day for a train during the observation period
#Input : Diagnostic data file for a train
#Output : Maximum number of alerts and critical alerts in a day during the observation period (e.g., 3 years observation period)
#----------------------------------------------------------------------------------------------------------------------
def GetMaxAlert(TrainData):
    maxalert = 0
    maxcriticalalert =0
    count_occ_critical=[]
    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    for datekey in TrainDataDiagosticMaintenance.keys():
        #print('date =  ', datekey, '   num alert = ', len(TrainDataDiagosticMaintenance[datekey]['Alertkey']), '  Critical = ', len(count_occ_critical))
        if (len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])>maxalert):
            maxalert = len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])

        count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[datekey]['AlertLevel']) if i == 1]
        if (len(count_occ_critical)>maxcriticalalert):
            maxcriticalalert = len(count_occ_critical)

    return maxalert, maxcriticalalert

#---------------------------------------------------------------------------------------------------------------------------
def FeatureExtraction(TrainData, alertthreshold, criticalalertthreshold):
    TrainMaintenanceDiagnosticInfoForAnTimeInterval = {}
    count_occ_critical=[]
    index=1
    numalert =0
    nummaintenance=0
    flagAlertAboveThreshold=0
    flagCriticalAlertAboveThreshold=0

    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    for datekey in TrainDataDiagosticMaintenance.keys():
        #print('date =  ', datekey, '   num alert = ', len(TrainDataDiagosticMaintenance[datekey]['Alertkey']), '  Critical = ', len(count_occ_critical))
        numalert = len(TrainDataDiagosticMaintenance[datekey]['Alertkey'])
        count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[datekey]['AlertLevel']) if i == 1]
        nummaintenance = TrainDataDiagosticMaintenance[datekey]['Maintenance']

        if (numalert>alertthreshold):
            flagAlertAboveThreshold = 1

        if (len(count_occ_critical)>criticalalertthreshold):
            flagCriticalAlertAboveThreshold = 1

        daysofweek = datekey.isoweekday() % 7
        #print('Days of the week = ', datekey.isoweekday() % 7)

        TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month': datekey.month, 'Year': datekey.year, 'DaysOfWeek': daysofweek,
                                                                  'Maintenance': nummaintenance, 'NumAlerts': numalert,
                                                                  'NumCriticalAlerts': len(count_occ_critical),
                                                                  'AlertAboveThreshold': flagAlertAboveThreshold,
                                                                  'CriticalAlertAboveThreshold': flagCriticalAlertAboveThreshold,
                                                                  }
        #print(TrainMaintenanceDiagnosticInfoForAnTimeInterval[index])
        index = index + 1

        flagAlertAboveThreshold=0
        flagCriticalAlertAboveThreshold =0

    return TrainMaintenanceDiagnosticInfoForAnTimeInterval

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

    #MaxAlertThreshold = maxalert * thresholdPercentageMaxAlertInADay
    #MaxCriticalAlertThreshold = maxcriticalalert * thresholdPercentageMaxAlertInADay
    max_avg_in_a_week=0
    max_avg_alert_in_a_week=0


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

            #if(len(TrainDataDiagosticMaintenance[datekey]['Alertkey']) >= MaxAlertThreshold): # check if the alert for this day exceeds of the threshold of max alert in  a day
            #    numdaywithmaxalert = numdaywithmaxalert+1
            #    alertabovemaxalert=1

            #if( len(count_occ_critical) >=  MaxCriticalAlertThreshold):
            #    numdaywithmaxcriticalalert = numdaywithmaxcriticalalert + 1
            #    alertabovemaxcriticalalert
        if(datecount>=intervaltime):
            #datekey.month, '  Year = ', datekey.year
            enddate = datekey
            if (numalert>0 and intervaltime>0):
                avgalert = numalert / intervaltime

            if(numcriticalalert>0 and intervaltime>0):
                avgcriticalalert = numcriticalalert/intervaltime

            print('avg alert = ', avgalert, '   avg critical alert = ', avgcriticalalert)
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
def ConvertTimeSeiresToSupervised(subdfTrainData):
    data = pd.DataFrame(subdfTrainData[['NumAlerts', 'NumCriticalAlerts', 'AlertAboveThreshold']].copy())

    # First : add raw lag data - add the lag of the target variable from 2 to 7 days
    for i in range(2, 7):
        data['Alertlag_{}'.format(i)] = data['NumAlerts'].shift(i)

    for i in range(2, 7):
        data['Criticallag_{}'.format(i)] = data['NumCriticalAlerts'].shift(i)

    lag_cols_alerr = [col for col in data.columns if 'Alertlag' in col]
    data['rolling_mean_Alert'] = data[lag_cols_alerr].mean(axis=1)

    lag_cols = [col for col in data.columns if 'Criticallag' in col]
    data['rolling_mean_Critical'] = data[lag_cols].mean(axis=1)


    # extract out the features and labels into separate variables (not dropping this raw feature increases performance)
    #y = data[label_col].values
    #data = data.drop('NumAlerts', axis=1)
    #data = data.drop('NumCriticalAlerts', axis=1)

    # drop rows with NaN values
    #if dropnan:
    data.dropna(inplace=True)

    X = data.values
    feature_names = data.columns
    print('dimension: ', X.shape)
    data.head()

    #print('Feature extraction = ', data.tail(10))
    return data


#----------------------------------------------------------------------------------------------------------
#Function : timeseries_train_test_split(X, y, test_size) -  Perform train-test split with respect to time series structure.
#----------------------------------------------------------------------------------------------------------
def timeseries_train_test_split(X, y, test_size):
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

#----------------------------------------------------------------------------------------------------------
#Function : LinearRegressionModel(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def LinearRegressionModel(trainX, trainy, testX, testy):
    model = LinearRegression()
    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    error = mean_absolute_percentage_error(prediction, testy)
    #print('Linear regression, MAE = ', error)
    #print('prediction = ', prediction)

    '''plt.figure(figsize=(15, 7))
    x = range(prediction.size)
    plt.plot(x, prediction, label='prediction', linewidth=2.0)
    plt.plot(x, testy, label='actual', linewidth=2.0)   
    plt.title('Mean absolute percentage error {0:.2f}%'.format(error)) 
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
'''
    return error
#----------------------------------------------------------------------------------------------------------
#Function : RandomForest(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def RandomForest(trainX, trainy, testX, testy):

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
    plt.xlabel('Number of Days')
    plt.ylabel('Number of alerts will exceed threshold')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return error
#----------------------------------------------------------------------------------------------------------
#Function : RandomForest(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def SupportVectorRegression(x_train, y_train,x_test,y_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train).reshape(-1, 1) #make prediction
    y_test_pred = model.predict(X_test).reshape(-1, 1)
    error = mean_absolute_percentage_error(y_test_pred.ravel(), y_test.ravel())

    y_test_pred[y_test_pred > 0.5] = 1
    y_test_pred[y_test_pred <= 0.5] = 0
    y_test_pred = y_test_pred.astype(int)

    '''
    plt.figure(figsize=(15, 7))
    x = range(y_test_pred.size)
    plt.plot(x, y_test_pred, label='prediction', linewidth=2.0)
    plt.plot(x, y_test, label='actual', linewidth=2.0)

    #plt.title('Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    '''
    return error
#-------------------------------------------------------------------------------------
def XGBRegressor(X_train,X_test, y_train, y_test):
    # create, train and do inference of the model
    #model = xgb.XGBRegressor()
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    #model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    error = mean_absolute_percentage_error(predictions, y_test)
    plt.title('Mean absolute percentage error {0:.2f}%'.format(error))

    plt.figure(figsize=(15, 7))

    x = range(predictions.size)
    plt.plot(x, predictions, label='prediction', linewidth=2.0)
    plt.plot(x, y_test, label='actual', linewidth=2.0)
    error = mean_absolute_percentage_error(predictions, y_test)
    plt.title('Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # create a dataframe with the variable importances of the model
    '''
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()
    '''

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
    #maxnumalertperday, maxnumcriticalalertperday = GetMaxAlert(TrainDataDiagosticMaintenance)
    #print('For this train, Max alert in a day =  ', maxnumalertperday, '   Max Critical Alert in a day =  ', maxcriticalalertcategoryperday)

    # sample data according to chosen slot
    TrainDataRecord = GetAlertforADuration(TrainDataDiagosticMaintenance, trainID, intervaltime,maxnumalertperday, maxnumcriticalalertperday)
    #print('Number of da ys in diagnostic file = ', len(TrainDataDiagosticMaintenance.keys()), '  Bin/sampling size (days)= ', intervaltime)

    #Define the threshold and extract features
    alertthreshold =300
    criticalalertthreshold = 3
    TrainDataFeture = FeatureExtraction(TrainDataDiagosticMaintenance, alertthreshold, criticalalertthreshold)

#-------------Processing of daily Time Series Data into a sample of week/bi-week/monthly slots and extracts features------------------------------------
    dfTrainData = pd.DataFrame.from_dict(TrainDataFeture, orient='index')
    subdfTrainData = dfTrainData[['Year', 'Month', 'DaysOfWeek', 'NumAlerts', 'NumCriticalAlerts', 'Maintenance','AlertAboveThreshold','CriticalAlertAboveThreshold']]  # Create new pandas DataFrame
    #print(subdfTrainData.tail(10))
    #print(subdfTrainData.info())
    TrainfeatureData = ConvertTimeSeiresToSupervised(subdfTrainData)
    #forecastWindow = 7  # data is stored in days, we want to forecast next 7 days
    outputFeatureName = 'AlertAboveThreshold'

    X = TrainfeatureData.drop(outputFeatureName, axis=1)
    y = TrainfeatureData[outputFeatureName]
    # Dividing into train and test set
    test_size=0.30
    X_train, X_test,y_train, y_test = timeseries_train_test_split(X, y, test_size)

#------------------Apply different ML models----------------------------------------------------------------------------------------------
    maelr = LinearRegressionModel(X_train, y_train, X_test, y_test)
    maerf = RandomForest(X_train, y_train, X_test, y_test)
    maesvr = SupportVectorRegression(X_train, y_train, X_test, y_test)
    #XGBRegressor(X_train, y_train, X_test, y_test)
    #print('X_test  before scaling = ', X_test)



    #print(len(y_train_pred), len(y_test_pred))

    print('----------------Mean absolute error for different ML algo ----------------')
    print('Linear Regression = ', maelr)
    print('Random Forest = ', maerf)
    print('Support Vector Regression = ', maesvr)
    exit(0)
    # ----------------------------------------------------------------------------------------------------------------
    # Feature extraction : convert time series into a format applicable for using Supervised Machine Learning techniques
    # Given this 3 dimensional time series data, we will need to do some feature-engineering to make it applicable for a
    # downstream supervised learning method. Including:
    # 1. Generating lagged features and window statistics from them.
    # 2. We will also label the output - if the alerts exceeds a threshold (boolean feature indicating whether the val exceeds a threshold).
    # ----------------------------------------------------------------------------------------------------------------

    #Random forest
    #LinearRegression()
    #SupportVectorRegression()



    #for i in range(len(test)):
# split test row into input and output columns


    #X_train = X[:test_index]
    #X_test = X[test_index:]
    #y_train = y[:test_index]
    #y_test = y[test_index:]

    #UnivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
    #MultivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
