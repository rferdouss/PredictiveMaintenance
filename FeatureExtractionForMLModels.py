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
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as mpl
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot


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
    subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts', 'Maintenance','AvgAlert','AvgCriticalAlert', 'NumDayWithMaxAlert', 'NumDayWithMaxCriticalAlert']]  # Create new pandas DataFrame


    print('data = ', subdfTrainData)
    varname = 'NumAlerts'  #'NumAlerts', 'NumCriticalAlerts'

    #UnivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
    #MultivariateTimeSeriesForecastingUsingLSTM(subdfTrainData, varname)
