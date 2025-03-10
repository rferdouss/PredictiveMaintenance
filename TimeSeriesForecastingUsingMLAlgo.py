#--------------------------------------------------------------------------------------------------------------
# File name : TimeSeriesForecastingUsingMLAlgo -
# Multi-variate time series single step forecasting problem - A python script to formulate a multi-variate time series forecasting problem into supervised machine learning problem
# Steps : Extract suitable features, apply various ML also, validate models

# Version : 1.0
# Raihana Ferdous
# Date :  01/07/2024  -- Started again : 01/01/2025
#--------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
import random
import math
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier  #Import RandomForestClassifier model model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score, classification_report  # Import evaluation metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV

from sklearn import svm  #Import svm model
from sklearn.naive_bayes import GaussianNB  #Import naive-bayes classifier model
from sklearn.neighbors import KNeighborsClassifier  # Import k-nearest-neighbor-classification
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.linear_model import LogisticRegression  # Import logistic regression

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from xgboost import XGBRegressor
#from fbprophet import Prophet

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import seaborn as sns
import tsfel
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor



#Input Output file location
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"

#Global variable
trainComp = 'TCU'
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"
outputfeaturelist = trainComp+"-Feature"

#-------------------------Global Variable need to define-------------------------------------------------------------------------
#Train information
trainname = 'TSR'
trainnumber = '070'  # T1 =070, T2 = 086, T3 =040, T4 = 008, T5=020 , T6=033, T7=, T8= ,T9=
traincomponent = 'TCU'

#smapling and feature engineering parameters - to be decided
intervaltime = 7 #Sampling the data   # weekly =7 , bi-weekly =15, monthly =30
AlertThresholdPercentageGlobal = 0.1 # value range = [0 -1], 0.70 means 70% of the max alert is considered
testdatasize =0.4  # for splitting the dataset into train and test set
NumberOfLagToBeConsider =8 # Number of previous data should be considered
featureflagall = True  # Whether to use all features or a set of most important features
#-----------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------
#   Extraction of features
#-------------------------------------------------------------------------------------------------------------------------
def FeatureExtraction(TrainData, alertthreshold, criticalalertthreshold):
    TrainMaintenanceDiagnosticInfoForAnTimeInterval = {}
    index=1
    flagAlertAboveThreshold=0
    flagCriticalAlertAboveThreshold=0

    #print(TrainData[1])

    #TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    for keyindex in TrainData.keys():
        #print('key ', TrainData[keyindex]['NumAlerts'])
        #print('date =  ', datekey, '   num alert = ', len(TrainDataDiagosticMaintenance[datekey]['Alertkey']), '  Critical = ', len(count_occ_critical))
        numalert = (TrainData[keyindex]['NumAlerts'])
        count_occ_critical = (TrainData[keyindex]['NumCriticalAlerts'])
        #count_occ_critical = [i for j, i in enumerate(TrainDataDiagosticMaintenance[keyindex]['AlertLevel']) if i == 1]
        nummaintenance = TrainData[keyindex]['Maintenance']

        if (numalert>alertthreshold):
            #print('Number of alert = ', numalert, '    Critical alert threshold = ', alertthreshold)
            flagAlertAboveThreshold = 1

        if ((count_occ_critical)>criticalalertthreshold):
            flagCriticalAlertAboveThreshold = 1

        #daysofweek = datekey.isoweekday() % 7
        #print('Days of the week = ', datekey.isoweekday() % 7)

        TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month': TrainData[keyindex]['Month'],
                                                                  'Year': TrainData[keyindex]['Year'],
                                                                  'WeekofYear': TrainData[keyindex]['WeekofYear'],
                                                                  'Maintenance': nummaintenance, 'NumAlerts': numalert,
                                                                  'NumCriticalAlerts': count_occ_critical,
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
def GetAlertforADuration(TrainData, trainID, intervaltime):
    TrainDataDiagosticMaintenance = ordered = OrderedDict(sorted(TrainData.items(), key=lambda t: t[0]))  # sort the dictionary based on datetime
    #print('Function GetAlertforADuration() - Get alert information for a time interval (days)= ', intervaltime)
    TrainMaintenanceDiagnosticInfoForAnTimeInterval={}

    datecount=1
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
    weekofYear=0

    #MaxAlertThreshold = maxalert * thresholdPercentageMaxAlertInADay
    #MaxCriticalAlertThreshold = maxcriticalalert * thresholdPercentageMaxAlertInADay
    max_alert_in_a_week=0
    min_alert_in_a_week=1000000
    max_criticalalert_in_a_week=0
    min_critical_alert_in_a_week=1000000
    max_avg_alert_in_a_week=0
    min_avg_alert_in_a_week=1000000

    avgalertarr=[]
    numalertarr=[]
    numcriticalalertarr=[]

    for datekey in TrainDataDiagosticMaintenance.keys():
        #print('datekey= ', datekey.weekofyear, 'datecount =', datecount)
        if(datecount ==1):
            startdate =  datekey
            weekofYear = datekey.isocalendar().week
            #print('week num',weekofYear)
            enddate = 0
        if(datecount<=intervaltime):
            nummaintenance= 0 #nummaintenance+ (TrainDataDiagosticMaintenance[datekey]['Maintenance']) #while considering only diagnostic data
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

            #get the max alert in a week
            if(numalert>max_alert_in_a_week):
                max_alert_in_a_week = numalert
            # get the min alert in a week
            if(numalert<min_alert_in_a_week):
                min_alert_in_a_week= numalert

            # get the max critical alert in a week
            if(numcriticalalert>max_criticalalert_in_a_week):
                max_criticalalert_in_a_week= numcriticalalert

            # get the min alert in a week
            if (numcriticalalert < min_critical_alert_in_a_week):
                min_critical_alert_in_a_week = numcriticalalert

            if(avgalert>max_avg_alert_in_a_week):
                max_avg_alert_in_a_week= avgalert

            # get the min alert in a week
            if (avgalert < min_avg_alert_in_a_week):
                min_avg_alert_in_a_week = avgalert


            #print('num alert = ', numalert, '   avg alert = ', avgalert, '   num critical alert = ', numcriticalalert,  '   avg critical alert = ', avgcriticalalert)

            #print('alert code = ', alertcode)
            #print('alert color =
            # Storing the data in an array for plotting
            numalertarr.append(numalert)
            numcriticalalertarr.append(numcriticalalert)
            avgalertarr.append(avgalert)

            TrainMaintenanceDiagnosticInfoForAnTimeInterval[index] = {'Month':datekey.month, 'Year': datekey.year,
                                                                      'WeekofYear':weekofYear,'StartDate':startdate,
                                                                      'EndDate':enddate,'Maintenance': nummaintenance,
                                                                      'NumAlerts': numalert, 'NumCriticalAlerts':numcriticalalert,
                                                                      'CriticalAlertCodeList':alertcode, 'CriticalAlertColor': alertcolor,
                                                                      'AvgAlert': avgalert, 'AvgCriticalAlert': avgcriticalalert,
                                                                      'AlertAboveThreshold': alertabovemaxalert,
                                                                      'CriticalAlertAboveThreshold':alertabovemaxcriticalalert,
                                                                      'NumDayWithMaxAlert': numdaywithmaxalert,
                                                                      'NumDayWithMaxCriticalAlert': numdaywithmaxcriticalalert}
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
    max_alert_in_a_week= round(max_alert_in_a_week, 2)
    min_alert_in_a_week = round(min_alert_in_a_week, 2)

    max_criticalalert_in_a_week = round(max_criticalalert_in_a_week, 2)
    min_avg_alert_in_a_week = round(min_avg_alert_in_a_week, 2)

    max_avg_alert_in_a_week = round(max_avg_alert_in_a_week, 2)
    min_avg_alert_in_a_week = round(min_avg_alert_in_a_week, 2)

    print('Size of the bin after sampling it with ',  intervaltime, '(days) interval = ', len(TrainMaintenanceDiagnosticInfoForAnTimeInterval.keys()))
    #print(TrainMaintenanceDiagnosticInfoForAnTimeInterval)
    print('Max weekly num alert = ', max_alert_in_a_week,   '   Min weekly alert =  ', min_alert_in_a_week)
    print('Max weekly num critical alert = ', max_criticalalert_in_a_week,   '   Min weekly critical alert =  ', min_critical_alert_in_a_week)
    print('Max weekly average alert = ', max_avg_alert_in_a_week,   '   Min weekly average alert =  ', min_alert_in_a_week)

    #plot  - num of alerts in a week
    ypointsAlert = np.array(numalertarr)
    fig, ax = plt.subplots(figsize=[8, 6])
    titlestr = trainID + "  - Histogram Num alert - Max weekly = " + str(max_alert_in_a_week) + "  Min weekly =  " + str(min_alert_in_a_week)

    ax.set_title(titlestr)  # set plot title
    ax.set_xlabel("Num of alert")    # set x-axis name
    ax.set_ylabel("frequency")    # set y-axis name
    N, bins, patches = ax.hist(ypointsAlert, bins=50, color="#777777")  # initial color of all bins
    plt.savefig(outputlocation+'\\Plot\\'+trainID+'\\Info\\AlertHist.png')
    #plt.show()

    # plot  - num of alerts in a week
    ypointscriticalAlert = np.array(numcriticalalertarr)
    fig, ax = plt.subplots(figsize=[8, 6])
    titlestr =trainID+ "  - Histogram Num Critical alert - Max weekly = " + str(max_criticalalert_in_a_week)+ "  Min weekly =  "+ str(min_critical_alert_in_a_week)
    ax.set_title(titlestr)  # set plot title
    ax.set_xlabel("Num of critical alert")  # set x-axis name
    ax.set_ylabel("frequency")  # set y-axis name
    N, bins, patches = ax.hist(ypointscriticalAlert, bins=50, color="#777777")  # initial color of all bins
    plt.savefig(outputlocation + '\\Plot\\' + trainID + '\\Info\\CriticalAlertHist.png')
    #plt.show()

    # plot  - num of alerts in a week
    ypointsavgAlert = np.array(avgalertarr)
    fig, ax = plt.subplots(figsize=[8, 6])
    titlestr = trainID + "  - Histogram Num Critical alert - Max weekly = " + str(max_avg_alert_in_a_week) + "  Min weekly =  " + str(min_avg_alert_in_a_week)
    ax.set_title(titlestr)  # set plot title
    ax.set_xlabel("Average num of alerts")  # set x-axis name
    ax.set_ylabel("frequency")  # set y-axis name
    N, bins, patches = ax.hist(ypointsavgAlert, bins=50, color="#777777")  # initial color of all bins
    plt.savefig(outputlocation + '\\Plot\\' + trainID + '\\Info\\AvgAlertHist.png')
    #plt.show()
    #store the features in a pickle file
    outputdataAnalysisfile = outputlocation + '\\' + trainID + '-' + outputfeaturelist
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))

    return TrainMaintenanceDiagnosticInfoForAnTimeInterval, max_alert_in_a_week, max_criticalalert_in_a_week, min_alert_in_a_week
#-----------------------------------------------------------------------------------------------------------------
# Feature extraction
#  Three types og features are extracted
#a)Date Time Features: these are components of the time step itself for each observation.
#b)Lag Features: these are values at prior time steps.
#c)Window Features: these are a summary of values over a fixed window of prior time steps.

#------------------------------------------------------------------------------------------------------------------
def ConvertTimeSeiresToSupervised(subdfTrainData, numberoflag):
    print('Convert time series into a Supervised learning problem')
    # First : add the original features ['AlertAboveThreshold' is the feature that we want to predict, so it is the y value, 'NumAlerts', 'NumCriticalAlerts' are the two time series given (x values)]

    if(featureflagall==True):
        print('Using all features ')
        data = pd.DataFrame(subdfTrainData[['Year', 'Month', 'WeekofYear', 'AlertAboveThreshold','NumAlerts', 'NumCriticalAlerts','Maintenance']].copy())
    #data = pd.DataFrame(subdfTrainData[['AlertAboveThreshold','NumAlerts', 'NumCriticalAlerts','Maintenance']].copy())
    else:
        print('Using a small set including only important features ')
        data = pd.DataFrame(subdfTrainData[['AlertAboveThreshold','NumAlerts', 'NumCriticalAlerts']].copy())

    # Second : add raw lag data - add the lag of the target variable from 1 to 4 weeks
    for i in range(1, numberoflag):
        data['Alertlag_{}'.format(i)] = data['NumAlerts'].shift(i)

    for i in range(1, numberoflag):
        data['Criticallag_{}'.format(i)] = data['NumCriticalAlerts'].shift(i)

    # Third : Rolling Window statistics (calculated from the lag values)
    lag_cols_alerr = [col for col in data.columns if 'Alertlag' in col]
    data['rolling_mean_Alert'] = data[lag_cols_alerr].mean(axis=1)

    lag_cols = [col for col in data.columns if 'Criticallag' in col]
    data['rolling_mean_Critical'] = data[lag_cols].mean(axis=1)

    lag_cols_alerr = [col for col in data.columns if 'Alertlag' in col]
    data['rolling_std_Alert'] = data[lag_cols_alerr].std(axis=1)

    lag_cols = [col for col in data.columns if 'Criticallag' in col]
    data['rolling_std_Critical'] = data[lag_cols].std(axis=1)

    # diff value
    #melt2['Last_Week_Diff'] = melt2.groupby(['Product_Code'])['Last_Week_Sales'].diff(


    # extract out the features and labels into separate variables (not dropping this raw feature increases performance)
    #y = data[label_col].values
    #data = data.drop('NumAlerts', axis=1)
    #data = data.drop('NumCriticalAlerts', axis=1)

    # drop rows with NaN values
    #if dropnan:
    data.dropna(inplace=True)

    X = data.values
    feature_names = data.columns
    #print('dimension: ', X.shape)
    data.head()

    #print('Feature extraction = ', data.head(10))
    return data

#----------------------------------------------------------------------------------------------------------
#Function : CalculateWeightedAverage(TrainDataRecord,max_alert_in_a_week, min_alert_in_a_week) -
# Calculate the weighted average of alerts for a duration
#----------------------------------------------------------------------------------------------------------
def CalculateWeightedAverage(TrainDataRecord,max_alert_in_a_week, min_alert_in_a_week):
    weightedAlertAvg=0
    binsize = len(TrainDataRecord.keys())
    binLength = round((max_alert_in_a_week - min_alert_in_a_week)/binsize)
    print('max =', max_alert_in_a_week, '   min= ', min_alert_in_a_week, '   Num of bin = ',binsize,  '  each bin length = ', binLength)
    weightvalue ={}
    startval = min_alert_in_a_week
    endval = min_alert_in_a_week + binLength
    frequency =0
    weight = 0

    #Step 1 : prepare the bin with the lenght of data
    for i in range(binsize+2):
        weightvalue[i]={'startval':startval, 'endval':endval,'freq':frequency, 'weight':weight}
        #print('Index = ',i, weightvalue[i])
        startval = endval + 1
        endval = (startval-1)+binLength

    #Step 2: populate the bins to calculate coverage
    for datakey in TrainDataRecord.keys():
        numalt =int(TrainDataRecord[datakey]['NumAlerts'])
        #print('Key value = ', TrainDataRecord[datakey]['NumAlerts'])
        for binkey in weightvalue.keys():
            #print(weightvalue[binkey]['startval'])
            if numalt >= int(weightvalue[binkey]['startval']):
                if numalt <= int(weightvalue[binkey]['endval']):
                    countoccureance = int(weightvalue[binkey]['freq'])+1
                    weightvalue[binkey]['freq'] = countoccureance
                    #print('num =', numalt, '   range =', weightvalue[binkey]['startval'], ' - ', weightvalue[binkey]['endval'], '  updated occurance = ', weightvalue[binkey]['freq'])
                    break

    #Step 3: Calculate the weight for each bin considering the occurance
    for binkey in weightvalue.keys():
        totoccurance = int(weightvalue[binkey]['freq'])
        if int(weightvalue[binkey]['freq']) > 0 :
            binweight = float(int(weightvalue[binkey]['freq'])/ len(weightvalue.keys()))
            weightvalue[binkey]['weight'] = round(binweight,4)
            #print('weight val = ', binweight)
        #print('index =', binkey, '   range =', weightvalue[binkey]['startval'], ' - ', weightvalue[binkey]['endval'], '  occurance = ', weightvalue[binkey]['freq'], '  Weight = ', weightvalue[binkey]['weight'])

    #Step4 : Finally  - Calculate the weighted average [Formula : weighted_average = sum(df['Values'] * df['Weights']) / sum(df['Weights'])]
    #print('Final Sept - Calculate Weighted Average :')
    sumValueWeight = 0
    sumWeight = 0
    weightedaverageValue=0
    for datakey in TrainDataRecord.keys():
        numalt =int(TrainDataRecord[datakey]['NumAlerts'])
        for binkey in weightvalue.keys():
            #print('Num alert = ', numalt)
            if numalt >= int(weightvalue[binkey]['startval']):
                if numalt <= int(weightvalue[binkey]['endval']):
                    bweight = float(weightvalue[binkey]['weight'])
                    valueweight = (numalt * bweight)
                    sumValueWeight = sumValueWeight+ valueweight
                    sumWeight = sumWeight +bweight
                    #print('Num alert for week = ', numalt, 'Found bin key index = ',  binkey,'  Weight  = ', bweight, 'Valueweight = ', valueweight, ' sumweight = ', sumWeight, '   sumvalweight = ', sumValueWeight)
                    break

    if(sumValueWeight > 0  and sumWeight > 0):  # weighted average value
        weightedAlertAvg = float(sumValueWeight/sumWeight)#the weighted average
    #print('Weighted average = ', weightedAlertAvg)
    return weightedAlertAvg

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
#Function : RandomWalk_BaselineModel() - Baseline classifier to compare the prediction results of ML algorithms
#----------------------------------------------------------------------------------------------------------
def RandomWalk_BaselineModel(Y_data,y_train, y_test, algoname, AlertThresholdPercentage, trainID):
    print('----------------------------------------------------------------------------------------------------------\n'
          'Baseline Model - Random walk Model\n'
          '-------------------------------------------------------------------------------------------------------------')
    actualtest =[]
    predicted=[]

    predictions = list()
    print('train data size= ', len(y_train), len(Y_data))
    #history = y_train[varname][len(train[varname])]
    history = y_train[len(y_train)]
    #print(len(y_train))
    #print('history = ', history, 'training set size = ', len(y_train))

    for i in range(len(Y_data)+1):
        if i > (len(y_train)):
            randomval = random.random()
            #print('i val = ', i, 'random val = ', randomval)
            if randomval<0.5:
                yhat = history -1
            else:
                yhat = history +1

            history = yhat

            if(yhat>0.5):
                yhat=1
            else:
                yhat=0
        #yhat = history + (-1 if np.random() < 0.5 else 1)
            #print('i val = ', i, 'random val = ', randomval, '  y predicted = ', history,'  y predicted final = ', yhat, '   y atual = ', Y_data[i])
            #print('test val = ', history, 'predicted = ', yhat, '   actual =', y_test[i])
            predicted.append(yhat)
            actualtest.append(Y_data[i])

    #print('out')
    print(actualtest, predicted)
    # get the prediction accuracy
    accuracy = accuracy_score(actualtest, predicted)
    precision = precision_score(actualtest, predicted)
    recall = recall_score(actualtest, predicted)
    f1 = 0#f1_score(predicted, actualtest, average="weighted")

    y_true = np.array(actualtest)
    sumvalue = np.sum(y_true)
    # mape = np.sum(np.abs((y_true - prediction))) / sumvalue * 100
    mape = mae(actualtest, predicted)
    print('Precision:', precision, '    Recall:', recall,'    Accuracy:', accuracy, '   F1 Score : ', f1  ,'   MeanAbsoluteError:', mape)

    # Generate classification report & store it using pickle dump
    classificaitonReport = classification_report(actualtest, predicted)
    #print('Classification Report = ', classificaitonReport)

    #error = np.square(np.subtract(actualtest,predicted)).mean()#mean_squared_error(test[varname], dfp)
    #rsmev = math.sqrt(error)
    #accuracy = 100 - (error*100) #accuracy_score(y_test, predicted)
    #print('Random WALK RMSE: %.3f' % rsmev, 'Accuracy: ', accuracy, '\n-----------------------------------------------------------')#, '  precision :', precision, '   recall:',recall, '  f1:',f1)

#--------#Random Forest forecasting algorithm-----------------------------------------------------------------
#Random Forest Classifier
# #Function : RandomForest(trainX, trainy, testX, testy)
#This function also performs hyperparameter tuning for a random forest classifier using random search. First, a dictionary param_dist is
#created with two hyperparameters to tune: n_estimators and max_depth. The values for these hyperparameters are randomly sampled from a
# uniform distribution between 50 and 500 for n_estimators and between 1 and 20 for max_depth. Next, a RandomForestClassifier object is
# created. Then, a RandomizedSearchCV object is created with the RandomForestClassifier object, the param_dist dictionary, and other
# parameters such as n_iter (the number of parameter settings that are sampled) and cv (the number of cross-validation folds to use).
# Finally, the RandomizedSearchCV object is fit to the training data (X_train and y_train) to find the best hyperparameters for the
# random forest classifier.

#----------------------------------------------------------------------------------------------------------
def RandomForestClassificationAlgo(trainX, trainy, testX, testy):
    print('Running Random Forest classifier........')
    #First do the hyperparameter tuning for getting the best model
    param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
    rf= RandomForestClassifier()  # Create a random forest classifier
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist,n_iter=5, cv=5) # Use random search to find the best hyperparameters
    rand_search.fit(X_train, y_train) # Fit the random search object to the data
    best_rf_model = rand_search.best_estimator_    # Create a variable for the best model
    print('Best hyperparameters:', rand_search.best_params_)
    prediction = best_rf_model.predict(X_test) # Generate predictions with the best model

    # Create a series containing feature importances from the model and feature names from the training data
    feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print('Random Forests - Feature Importance result = ', feature_importances.values)
    feature_importances.plot.bar() # Plot a simple bar chart
    plt.title('Random Forest Classifier - Feature Importance')
    plt.xlabel('Features')
    plt.ylim(0,(max(feature_importances.values)+0.2))
    plt.savefig(outputlocation+'\\Plot\\'+trainID+'\\RandomForest-FeatureImportance.png')
    #plt.show()
    return prediction

#----------------------------------------------------------------------------------------------------------
#Function : SupportVectorMachine(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def SupportVectorMachine(X_train, y_train, X_test, y_test):
    print('Running Support Vector Machine........')
    model = svm.SVC(kernel='linear')  # Linear Kernel  # Create a svm Classifier
    model.fit(X_train, y_train)   # Train the model using the training sets
    prediction = model.predict(X_test) # Predict the response for test dataset
    return prediction

# ----------------------------------------------------------------------------------------------------------
# AlgoName : Logistic Regression - A supervised machine learning algorithm that accomplishes binary classification
# tasks by predicting the probability of an outcome, event, or observation
# Function : LogisticRegression(trainX, trainy, testX, testy)
# ----------------------------------------------------------------------------------------------------------
def LogisticRegressionModel(X_train, y_train, X_test, y_test):
    print('Running Logistic Regression ...............')
    model = LogisticRegression(random_state=16)  # instantiate the model (using the default parameters)
    model.fit(X_train, y_train) # fit the model with data
    prediction = model.predict(X_test)
    return prediction

#----------------------------------------------------------------------------------------------------------
#Function : Naive-Bayes(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def KNearestNeighborsClassifier(X_train, y_train, X_test, y_test):

    scaler = StandardScaler()  # Scale the features using StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    #nsamples, nx, ny = X_train.shape
    #d2_X_train_dataset = X_train.reshape((nsamples, nx * ny))



    #model = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=10)
    model = KNeighborsTimeSeriesClassifier(distance="euclidean")

    model.fit(X_train, y_train)
    #nsamples, nx, ny = train_dataset.shape
    #d2_train_dataset = train_dataset.reshape((nsamples, nx * ny))
    

    prediction = model.predict([X_test]) # Predict Output
    print('k neighbor')

    print('prediction = ', prediction)
    print('Actual Data = ', y_test)

    print("Actual Value:", y_test)
    print("Predicted Value:", prediction)

    # get the prediction accuracy
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(prediction, y_test, average="weighted")
    error = mean_absolute_percentage_error(prediction, y_test)

    print("Precision:", precision)
    print("Recall:", recall)
    print("built in funciton Accuracy:", accuracy)

    plt.figure(figsize=(15, 7))
    x = range(prediction.size)
    plt.plot(x, prediction, label='prediction', linewidth=2.0)
    plt.plot(x, y_test, label='actual', linewidth=2.0)
    plt.title('KNearest Neighbor - Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    #hyper parameter tunning (need to do it with walk through validation instead of cross-fold validation)

    k_values = [i for i in range(1, 31)]
    scores = []

    scaler = StandardScaler()
    X = 0#scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = 0#cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))
        
    best_index = np.argmax(scores)
    best_k = k_values[best_index]

    knn = KNeighborsClassifier(n_neighbors=best_k) # retraining with the best k val
    knn.fit(X_train, y_train)

    return error
#----------------------------------------------------------------------------------------------------------
#Function : Naive-Bayes(trainX, trainy, testX, testy)
#----------------------------------------------------------------------------------------------------------
def NaiveBayesClassification(X_train, y_train, X_test, y_test):
    model = GaussianNB()      # Build a Gaussian Classifier
    model.fit(X_train, y_train)      # Model training

    #nsamples, nx, ny = train_dataset.shape
    #d2_train_dataset = train_dataset.reshape((nsamples, nx * ny))


    prediction = model.predict([X_test]) # Predict Output

    print('prediction = ', prediction)
    print('Actual Data = ', y_test)

    print("Actual Value:", y_test)
    print("Predicted Value:", prediction)

    # get the prediction accuracy
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(prediction, y_test, average="weighted")
    error = mean_absolute_percentage_error(prediction, y_test)

    print("Precision:", precision)
    print("Recall:", recall)
    print("built in funciton Accuracy:", accuracy)

    plt.figure(figsize=(15, 7))
    x = range(prediction.size)
    plt.plot(x, prediction, label='prediction', linewidth=2.0)
    plt.plot(x, y_test, label='actual', linewidth=2.0)
    plt.title('NaiveBayes - Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return error

#--------------------------------------------------------------------------------------
def XGBRegressorForecasting(X_train, y_train, X_test, y_test):
    #print('label size', X_train)
    #print(y_train)
    # create, train and do inference of the model
    model = XGBRegressor(n_estimators=1000)
    #model = XGBRegressor(objective= 'multi:softmax', n_estimators=1000)
    #model = MultiOutputRegressor(
    #    estimator=XGBRegressor()
    #)

    model.fit(X_train, y_train)
    #model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    # Add 'Even' for even numbers, otherwise 'Odd'
    #result = [1 if n >= 0.8 else 0 for n < 0.8]
    print('prediction = ', predictions)
    print('actual = ', y_test)

    error = mean_absolute_percentage_error(predictions, y_test)

    plt.figure(figsize=(15, 7))
    x = range(predictions.size)
    plt.plot(x, predictions, label='prediction', linewidth=2.0)
    plt.plot(x, y_test, label='actual', linewidth=2.0)
    plt.title('XGBRegression - Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return error

'''
def walk_forward_validation(data, percentage=0.2):
# In this case -1 is the target column (last one)
train, test = train_test_split(data, percentage)
predictions = []
history = [x for x in train]

for i in range(len(test)):
    test_X, test_Y = test[i, :-1], test[i, -1]
    pred = xgb_prediction(history, test_X)
    predictions.append(pred)
    history.append(test[i])

Y_test = target_scaler.inverse_transform(test[:, -1:].reshape(1, -1))
Y_pred = target_scaler.inverse_transform(np.array(predictions).reshape(1, -1))
test_rmse = mean_squared_error(Y_test, Y_pred, squared=False)  # squared=False to get RMSE instead of MSE

return test_rmse, Y_test, Y_pred
'''
#-------------Validation of algorithms-------------------------------------------------------------
def AlgorithmValidation(alertthresholdpercentage, y_prediction, y_actual, algoname, trainid):
    #alertthresholdpercentageval =alertthresholdpercentage*100
    #print('prediction = ', y_prediction)
    #print('Actual Data = ', y_actual)

    # get the prediction accuracy
    accuracy = accuracy_score(y_actual, y_prediction)
    precision = precision_score(y_actual, y_prediction)
    recall = recall_score(y_actual, y_prediction)
    f1 = f1_score(y_prediction, y_actual, average="weighted")

    y_true = np.array(y_actual)
    sumvalue = np.sum(y_true)
    #mape = np.sum(np.abs((y_true - prediction))) / sumvalue * 100
    mape = mae(y_actual, y_prediction)
    #print('Precision:', precision, '    Recall:', recall,'    Accuracy:', accuracy, '   F1 Score : ', f1  ,'   MeanAbsoluteError:', mape)

    # Generate classification report & store it using pickle dump
    classificaitonReport = classification_report(y_actual, y_prediction)
    #print('Classification Report = ', classificaitonReport)
    classificationresultpath = outputlocation + '\\Plot\\' + trainID +'\\ClassificationReport'
    pickle.dump(classificaitonReport, open(classificationresultpath, "wb"))

    # Create and plot the confusion matrix
    cm = confusion_matrix(y_actual, y_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    #plt.savefig(outputlocation + '\\Plot\\' + trainid + '\\' + algoname + '_AlertThresholdPercentage' + str(alertthresholdpercentage) + '_ConfusionMatrix.png')
    plt.savefig(outputlocation + '\\Plot\\' + trainid + '\\' + algoname + '_ConfusionMatrix.png')

# plot actual and predicted value
    plt.figure(figsize=(15, 7))
    x = range(y_prediction.size)
    plt.plot(x, y_prediction, label='prediction', linewidth=2.0)
    plt.plot(x, y_actual, label='actual', linewidth=2.0)

    plt.title(algoname+ ' - Mean absolute percentage error {0:.2f}%'.format(mape))
    plt.legend(loc='best')
    plt.xlabel('Number of Days')
    plt.ylabel('Number of alerts will exceed threshold')
    plt.tight_layout()
    plt.grid(True)
    #plt.show()
    #plt.savefig(outputlocation+'\\Plot\\'+trainid+'\\'+algoname+'_AlertThresholdPercentage'+str(alertthresholdpercentage)+'_Prediction.png')
    plt.savefig(outputlocation+'\\Plot\\'+trainid+'\\'+algoname+'_Prediction.png')
    return precision, recall, accuracy, f1, mape

#---------------------------------------------------------------------------------------------------
#  Function that run different ML algorithm on time series data (that are converted into a supervised
#  learning problem) and perform validation
#  Function : TimeSeriesForecastingWithMLAlgo(X_train, y_train, X_test, y_test, algoname, AlertThresholdPercentage)
# MLAlgoName=['Random Forest', 'Support Vector Machine','Logistic Regression']
#-----------------------------------------------------------------------------------------------------
def TimeSeriesForecastingWithMLAlgo(X_train, y_train, X_test, y_test, algoname, AlertThresholdPercentage,trainID):
    print('\n\n------------------Starting Time series forecasting using ML algorithms -------------')
    if (algoname =='Random Forest'):  # **. Random Forest Classification Algorithm
        y_prediction = RandomForestClassificationAlgo(X_train, y_train, X_test, y_test)

    if (algoname =='Support Vector Machine'):# **. Support Vector Machine - Classification Algorithm
        y_prediction = SupportVectorMachine(X_train, y_train, X_test, y_test)

    if (algoname =='Logistic Regression') : # **. Logistic Regression' - Classification Algorithm
        y_prediction = LogisticRegressionModel(X_train, y_train, X_test, y_test)

    # Validation - Evaluate the effectiveness of the algo
    precision, recal, accuracy, f1score, mape = AlgorithmValidation(AlertThresholdPercentage, y_prediction, y_test,algoname, trainID)

    return precision, recal, accuracy, f1score, mape


#------------------------------------------------------------------------------------
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainDiagnosticfilename =dataInputLocation +"\\DiagnosticData\\"+trainname+" "+trainnumber+".csv"
    trainMaintenanceDatafilename = dataInputLocation +"\\MaintenanceData\\Avvisi SAP - Interventi manutentivi.xlsx"
    datasheetname =trainname+" "+trainnumber
    trainID = trainname+trainnumber


#----------------------------------------------------------------------------------------------------------------
# Load processed data that are stored in pickle files
#----------------------------------------------------------------------------------------------------------------
    print('Loading processed data file for prediction analysis : ')
    #outputpicklefilename= outputlocation + '\\'+ trainID + '-'+outputfname # diagnostic and maintenance data
    outputpicklefilename= outputlocation + '\\'+ trainID + '-'+outputdiagnosticfname # only diagnostic data
    print(outputpicklefilename)
    TrainDataDiagosticMaintenance = LoadProcessedTrainData(outputpicklefilename)
    #print(TrainDataDiagosticMaintenance)

# ----------------------------------------------------------------------------------------------------------------
# Analysis the data and sampling
# ----------------------------------------------------------------------------------------------------------------
    # sample data according to chosen slot
    TrainDataRecord, max_alert_in_a_week, max_criticalalert_in_a_week, min_alert_in_a_week = GetAlertforADuration(TrainDataDiagosticMaintenance, trainID, intervaltime)
    weightedAvgNumAlert = CalculateWeightedAverage(TrainDataRecord,max_alert_in_a_week, min_alert_in_a_week)
    print('Weighted Average = ', weightedAvgNumAlert)

# ----------------------------------------------------------------------------------------------------------------
#  Extracting the feature, Run the experiments varying the Threshold level for alerts
# ----------------------------------------------------------------------------------------------------------------
    MLAlgoName=['Random Forest', 'Support Vector Machine','Logistic Regression']
    #MLAlgoName=['Random Forest']
    MLPredictionModelEvaluation = {}  # dictionary to store the validation results obtained from different ML algo

    for algoname in MLAlgoName:  # Iterate over the list of algorithms
        print('---------------------------------------------------------------------------')
        AlertThresholdPercentage= AlertThresholdPercentageGlobal
        print('Algo Name = ', algoname)
        while (AlertThresholdPercentage <= 0.6):  # Iterate over a range of alert threshold for each algorithm
            #Define the threshold and extract features
            #--------------------------------
            #AlertThresholdPercentage=round(AlertThresholdPercentage,2)
            #alertthreshold = round((max_alert_in_a_week * AlertThresholdPercentage), 2)
            #criticalalertthreshold = round((max_criticalalert_in_a_week * AlertThresholdPercentage), 2)
            #-----------------------------

            alertthreshold =  weightedAvgNumAlert
            criticalalertthreshold=0
            print('Alert Threshold Percentage = ', AlertThresholdPercentage, '   Alert Threshold = ', alertthreshold, '   Critical alert threshold = ', criticalalertthreshold)
            #TrainDataFeture = FeatureExtractionFromPerDayData(TrainDataDiagosticMaintenance, alertthreshold, criticalalertthreshold)  # Extract suitable feature
            # Feature extraction from sampled (weekly, biweekly, monthly sampled bin) data
            TrainDataFeture = FeatureExtraction(TrainDataRecord, alertthreshold, criticalalertthreshold)  # Extract suitable feature

            #-------------Processing of daily Time Series Data into a sample of week/bi-week/monthly slots and extracts features------------------------------------
            dfTrainData = pd.DataFrame.from_dict(TrainDataFeture, orient='index')
            subdfTrainData = dfTrainData[['Year', 'Month', 'WeekofYear', 'NumAlerts', 'NumCriticalAlerts', 'Maintenance','AlertAboveThreshold','CriticalAlertAboveThreshold']]  # Create new pandas DataFrame
            #print(subdfTrainData.tail(10))
            #print(subdfTrainData.info())
            numberoflagcounter=NumberOfLagToBeConsider  # number of lag or previous value will be consider
            TrainfeatureData = ConvertTimeSeiresToSupervised(subdfTrainData, numberoflagcounter)


            #forecastWindow = 7  # data is stored in days, we want to forecast next 7 days
            outputFeatureName = 'AlertAboveThreshold'

            X = TrainfeatureData.drop(outputFeatureName, axis=1)
            y = TrainfeatureData[outputFeatureName]

            # ----------------------------------------------------------------------------------------------------------------
        # # Dividing into train and test set
        # ----------------------------------------------------------------------------------------------------------------
            test_size=testdatasize
            X_train, X_test,y_train, y_test = timeseries_train_test_split(X, y, test_size)

            print('Train set size =', len(y_train), '  Test set size = ', len(y_test))
            #print('Train ', y_test)
            # Compare with the baseline random walk model
            RandomWalk_BaselineModel(y,y_train, y_test, algoname, AlertThresholdPercentage, trainID) # BaseLine algorithm


        # ----------------------------------------------------------------------------------------------------------------
        #  Hyperparameter Tuning - for different ML algo
        # ----------------------------------------------------------------------------------------------------------------
            #RandomForest_HyperparameterTunning(X_train, y_train)

        #------------------------------------------------------------------------------------------------------------
            # Apply different ML supervised classification algorithms
        #---------------------------------------------------------------------------------------------
            #KNearestNeighborsClassifier(X_train, y_train, X_test, y_test)
            #NaiveBayesClassification(X_train, y_train, X_test, y_test)

            # Run time series forecasting with differnt ML algorithms and collect results
            precision, recall, accuracy, f1score, mape = TimeSeriesForecastingWithMLAlgo(X_train, y_train, X_test, y_test, algoname, AlertThresholdPercentage, trainID)

            #Store the result in a dictionary
            dickey = algoname+ '-AlertThreshold'+ str(AlertThresholdPercentage)
            if dickey not in MLPredictionModelEvaluation.keys():
                MLPredictionModelEvaluation[dickey] = {'Accuracy':accuracy,'F1Score':f1score,'Precision ':precision, 'Recall':recall, 'MeanAbsoluteEerror':mape}



            AlertThresholdPercentage =AlertThresholdPercentage+ 0.1  # Increase the AlertThreshold
            break
            #print(len(y_train_pred), len(y_test_pred))



    #Finally printing and storing the results
    print('------------------Validation Result-------------------------', MLPredictionModelEvaluation.keys())
    for key in MLPredictionModelEvaluation.keys():
        print('Algo Name and Alert threshold = ',key, '  Validation Result =  ', MLPredictionModelEvaluation[key])
        #print('Algo Name and Alert threshold = ', key, '  Accuracy = ', MLPredictionModelEvaluation[key]['Accuracy'],
        #      '  Precision = ', MLPredictionModelEvaluation[key]['Precision'], '  Recall = ', MLPredictionModelEvaluation[key]['Recall'],
        #      '  F1Score = ', MLPredictionModelEvaluation[key]['F1Score'], '  MeanAbsoluteEerror = ', MLPredictionModelEvaluation[key]['MeanAbsoluteEerror']
        #      )
    # store validation result in a file
    resultpath = outputlocation + '\\Plot\\' + trainID +'\\Result'
    pickle.dump(MLPredictionModelEvaluation, open(resultpath, "wb"))
