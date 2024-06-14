#======================================================================================================================
# DataProcessingTrenordDiagnosticMaintenanceRecords.py -
# Description :  This script is usable on the shared data from Trenord from May 2024
# (To make this code applicable on the previous data set share in November 2023, we need to
# convert the following headers on the raw csv file)
# This python script is to process and clean the diagnostic and maintenance data of Trenord (From July 2020 - March 2023)

# Input : A .csv file containing diagnostic record of a train
# Output: Python dictionary

# Raihana Ferdous
# Date : 29/05/2024
#==========================================================================================================================
import math
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl


#---------------------------Global variables---------------------------------------------------------------------------------
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'   # considered train component
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"

listofPDMOnly=['5-17--1','5-3--1', '5-5--1','5-7--1', '5-9--1','5-34--1','5-35--1','5-36--1']  # For TCU - List of PDM alert that are not associated with any PDO
#orphan pdm ='5-17--1','5-8--1', '5-6--1','5-4--1', '5-2--1'
#------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
# Function : ProcessDiagnosticMaintenanceData
# This function is used to process both diagnostic and maintenance data and put them in a python dictionary
# Important features of DiagnosticData : name, alert_type, cod, PdM, PdO, event_type, depot
# discard data if 'depot' == 1
#-----------------------------------------------------------------------------------------------------------------------------
def ProcessDiagnosticMaintenanceData(df_DiagnosticData,df_MaintenanceData,AlertCode, trainID, trainCarList):
    TrainData={}
    ListMissingAlermCode=[]
    CountPDMwithoutPDO=0
    considercomponentcode=0
    numofTCUDiganosticeventbeforefiltering=0
    Type =""

    if (trainComp =='TCU'):
        considercomponentcode=5  # 5 indicate TCU unit
        print('Analysis TCU data')

    if (trainComp =='DCU'):
        considercomponentcode=7  # 7 indicate TCU unit
        print('Analysis DCU data')
    print('================Start reading raw diagnostic event from file---------------------------------')
    numrecord=0
   # iterate through each row of the diagnostic data
    for ind in df_DiagnosticData.index:
        numrecord=numrecord+1
        sametraindata=0
        newRecordinDictionary = 0  # 1= new entry, 0= entry exists, indicates if this entry already exists in the dictionary
        duplicatealert = 0  # 1= found duplicate, 0 = new alert, check if an alert for a timestamp is already stored in a dictionary

        if df_DiagnosticData['source'][ind] in trainCarList:  # check whether all diagnostic events are from the same train
            sametraindata=1

        if df_DiagnosticData['cod'][ind]==considercomponentcode and sametraindata==1:  # consider TCU records only
            numofTCUDiganosticeventbeforefiltering= numofTCUDiganosticeventbeforefiltering+1
            #print(' Data group = ',  df_DiagnosticData['cod'][ind])
            #print('---Event Index = ', ind,  '   type = ', df_DiagnosticData['alert_type'][ind], 'deposit = ', df_DiagnosticData['depot'][ind], '  event = ', df_DiagnosticData['event_type'][ind])
            if df_DiagnosticData['depot'][ind]!=1 and df_DiagnosticData['event_type'][ind]=='ON':   # discard rows with df['depot'][ind]==1, as they are not relevant, reference dataprocessing notes from Trenord
                #print('deposit = ', df_DiagnosticData['depot'][ind], '  event = ', df_DiagnosticData['event_type'][ind])
                #print('Time of event = ', df_DiagnosticData['ts'][ind], '   Car Num= ', df_DiagnosticData['name'][ind])
                # print('pdm = ', str(df_DiagnosticData['PdM'][ind]).split()[0],'   pdo = ',   str(df_DiagnosticData['PdO'][ind]))


    #--------------Step 1---------------------------------------------------------------------------------------------------
    #   First check whether it is a PDM or PDO, we usually go for PDO, only cosider a PDM if there is no associated PDO with it
    #-----------------------------------------------------------------------------------------------------------------------
                #print('Group = ', str(df_DiagnosticData['cod'][ind]).split()[0], '   PDM = ',str(df_DiagnosticData['PdM'][ind]).split()[0], '   PDO =', str(df_DiagnosticData['PdO'][ind]).split()[0])
                # if it is of type other than PDO and PDM (e.g., trainset) then discard
                if (df_DiagnosticData['alert_type'][ind] =='TrainSet'):
                    print('Trainset - maintenance event - continue')
                    continue

                else:  # go forward if it is a PDO or PDM
                    if(int(str(df_DiagnosticData['PdO'][ind]).split()[0])==0):
                        Type ='PDM'
                        Alertkey = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdM'][ind]).split()[0], str('-1'))

                        if Alertkey in listofPDMOnly:  # only consider the PDM that is not associated with any PDO, otherwise discard
                            # print('PDM only = ', Alertkey, '    listpdmonly = ', listofPDMOnly)
                            CountPDMwithoutPDO = CountPDMwithoutPDO + 1
                        else:
                            #print('PDM --discarding')
                            continue
                    else:
                        Type = 'PDO'
                        Alertkey = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdM'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdO'][ind]).split()[0])
                        # print('PDO=  ', Alertkey)


                #print('Type original = ', df_DiagnosticData['alert_type'][ind],'   Type after investigation = ', Type, '  alertkey = ', Alertkey)


    # --------------Step 2---------------------------------------------------------------------------------------------------
    #   If we are fine with the alert code, then we go for making an entry, check if this alert is an duplicate one
    # -----------------------------------------------------------------------------------------------------------------------
                if df_DiagnosticData['ts'][ind] not in TrainData.keys():
                    newRecordinDictionary = 1  # indicates that this event is a new event with a new timestamp
                    #print('newRecordinDictionary = ', newRecordinDictionary, 'It is a New entry=', df_DiagnosticData['ts'][ind])
                    TrainData[df_DiagnosticData['ts'][ind]] = {'Alertkey':[], 'AlertColor':[], 'AlertLevel':[],
                                                                               'CriticalAlertCode':[], 'CriticalAlertColor':[],
                                                                               'NumPDO':0, 'NumPDM':0,'NumAlert':0,'NumCritical':0,
                                                                               'Diagnostic': 1,'MaintenanceDesc':[],'MaintenanceDescDet':[],
                                                                               'TypeMaintenance':[],'EndMaintenance':[], 'Maintenance':0,
                                                                               'machine_type':[],'source':[],'name': [], 'alert_type': [],
                                                                                'cod': [],'PdM': [], 'PdO': [], 'event_type': [],
                                                                                'lat': [],'lon': []}



                #Alertkey = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(df_DiagnosticData['PdO'][ind]).split()[0])
                #print('Date : ', df_DiagnosticData['ts'][ind], '   Alert type = ', df_DiagnosticData['alert_type'][ind], '  index = ', ind, '  Alert key in diagnostic data = ', Alertkey)

                # if it is not a new entry then check if this is a duplicate alert, if so then discard otherwise go forward for storing it into the dictionary
                #if(newRecordinDictionary == 0):  # if an entry already exists
                #    print('An entry already exists for this timestamp = ', df_DiagnosticData['ts'][ind])
                #    duplicatealert = CheckForDuplicateAlertKey(TrainData, df_DiagnosticData['ts'][ind], df_DiagnosticData['source'][ind], df_DiagnosticData['name'][ind], Alertkey, df_DiagnosticData['machine_type'][ind])

                # Check for the alertkey of the alert event into alert database
                if '0-0-0' not in Alertkey and duplicatealert==0:  # duplicatealert=0 indicate no prev entry for this alert is already stored in this timestamp
                    #print('False Alerm')
                    if((Alertkey in AlertCode.keys())):
                        #print('AlertKey found')
                        TrainData[df_DiagnosticData['ts'][ind]]['Alertkey'].append(Alertkey)
                        TrainData[df_DiagnosticData['ts'][ind]]['AlertColor'].append(AlertCode[Alertkey]['Colore'])
                        TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel'].append(AlertCode[Alertkey]['Critico'])
                        if (AlertCode[Alertkey]['Critico']==1):
                            TrainData[df_DiagnosticData['ts'][ind]]['CriticalAlertCode'].append(Alertkey)
                            TrainData[df_DiagnosticData['ts'][ind]]['CriticalAlertColor'].append(AlertCode[Alertkey]['Colore'])

                    else:
                        print('-----------------------Alert key not found = ', Alertkey)
                        if (str(df_DiagnosticData['PdO'][ind]) == '0'):
                            AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(float("nan")))
                        else:
                            if(pd.isna(float(str(df_DiagnosticData['PdO'][ind]))) == True):
                                AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],
                                                                str(df_DiagnosticData['PdM'][ind]).split()[0],
                                                                str('0'))
                            #print('Alternate alert key = ', AlertkeyAlternate)
                            if(AlertkeyAlternate in AlertCode.keys()):
                                #print('Alternate alert key found in Alerm list = ', AlertkeyAlternate)
                                TrainData[df_DiagnosticData['ts'][ind]]['Alertkey'].append(AlertkeyAlternate)
                                TrainData[df_DiagnosticData['ts'][ind]]['AlertColor'].append(AlertCode[AlertkeyAlternate]['Colore'])
                                TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel'].append(AlertCode[AlertkeyAlternate]['Critico'])
                            else:
                                print('Finally Alternate alerm code is not found = ', AlertkeyAlternate, 'orignal key=  ',Alertkey)
                                ListMissingAlermCode.append(AlertkeyAlternate)
                else:
                    print('False Alerm = ', Alertkey)
                    pass


                TrainData[df_DiagnosticData['ts'][ind]]['name'].append(df_DiagnosticData['name'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['alert_type'].append(df_DiagnosticData['alert_type'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['cod'].append(df_DiagnosticData['cod'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['PdM'].append(df_DiagnosticData['PdM'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['PdO'].append(df_DiagnosticData['PdO'][ind])
                #print('To check = ', (np.count_nonzero(TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel'])), "      = ", TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel'])
                #Calculated value
                TrainData[df_DiagnosticData['ts'][ind]]['NumPDO']=len(TrainData[df_DiagnosticData['ts'][ind]]['PdO'])
                TrainData[df_DiagnosticData['ts'][ind]]['NumPDM']=len(TrainData[df_DiagnosticData['ts'][ind]]['PdM'])


                TrainData[df_DiagnosticData['ts'][ind]]['NumAlert']=len(TrainData[df_DiagnosticData['ts'][ind]]['Alertkey'])
                TrainData[df_DiagnosticData['ts'][ind]]['NumCritical']=(np.count_nonzero(TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel']))

                TrainData[df_DiagnosticData['ts'][ind]]['event_type'].append(df_DiagnosticData['event_type'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['lat'].append(df_DiagnosticData['lat'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['lon'].append(df_DiagnosticData['lon'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['source'].append(df_DiagnosticData['source'][ind])
                TrainData[df_DiagnosticData['ts'][ind]]['machine_type'].append(df_DiagnosticData['machine_type'][ind])

    #print(len(uniqueEntry))
    #print(uniqueEntry)
    print('Diagnostic Data Summary')
    print('---------------------------------------------------------------')
    print('Number of PDMwithoutPDO = ', CountPDMwithoutPDO)
    print('Total diagnostic event (TCU+DCU) recorded in the file = ', numrecord)
    print('Total TCU diagnostic events in the file (before filtering) = ', numofTCUDiganosticeventbeforefiltering)
    print('Number of days where diagnostic events are found = ', len(TrainData.keys()))
    #print(TrainData)

    #print('Size of Data dictionary  ', len(TrainData.keys()))
    print('Missing alerm code = ', len(ListMissingAlermCode))
    #print(ListMissingAlermCode)
    print('List of days =  ', TrainData.keys())

  #---------------------------------------------------------------------------------------------------------------------------------
  #  Step 3 : Take the half of the alerts and critical alerts to discard the duplicate PDO (as in theory each PDO appears twice in the event log)
  #----------------------------------------------------------------------------------------------------------------------------------
    totalalertfortrain=0
    totalcriticalalertfortrain=0
    for tkey in TrainData.keys():
        #print('Key = ', tkey)
        befalert = TrainData[tkey]['NumAlert']
        befcriticalalert= TrainData[tkey]['NumCritical']
        if( TrainData[tkey]['NumAlert'] >0):
            TrainData[tkey]['NumAlert'] = math.ceil(TrainData[tkey]['NumAlert'] /2)
            totalalertfortrain=totalalertfortrain+ TrainData[tkey]['NumAlert']

        if( TrainData[tkey]['NumCritical']>0):
            TrainData[tkey]['NumCritical'] = math.ceil(TrainData[tkey]['NumCritical']/2)
            totalcriticalalertfortrain=totalcriticalalertfortrain+TrainData[tkey]['NumCritical']
        #print('Key = ', tkey, '   Bef alert = ',befalert, '    Bef Num critical = ',befcriticalalert,   'Num alerts = ', TrainData[tkey]['NumAlert'], '   Critical = ', TrainData[tkey]['NumCritical'])
        #print('Key = ', tkey, '   num alert = ', TrainData[tkey]['NumAlert'], '   Num critial alert = ', TrainData[tkey]['NumCritical'])

    print('Number of alerts (after all filtering) in the file = ', totalalertfortrain)
    print('Number of critical alerts (after all filtering) in the file = ', totalcriticalalertfortrain)


#=========================================================================================================================================
#                           Start processing maintenance data
#==========================================================================================================================================
    # Maintenance Data : iterate through each row of the maintenance data
    print('================Start reading raw maintenance event from file---------------------------------')
    storedate = []
    nummaintenanceevent = 0
    totalmaintenanceevent=0
    for ind in df_MaintenanceData.index:
        totalmaintenanceevent=totalmaintenanceevent+1
        if(df_MaintenanceData['Descr. assem. Padre'][ind]=='Trazione'):
            nummaintenanceevent=nummaintenanceevent+1
            #print('Maintenance of tranction devices')
            if df_MaintenanceData['Inizio guasto'][ind] not in TrainData.keys():  # if already an entry exists for storing the diagnostic data for this day
                TrainData[df_MaintenanceData['Inizio guasto'][ind]] = {'Alertkey': [], 'AlertColor': [],
                                                                           'AlertLevel': [],
                                                                           'CriticalAlertCode': [],
                                                                           'CriticalAlertColor': [],
                                                                           'NumPDO': 0, 'NumPDM': 0, 'NumAlert': 0,
                                                                           'NumCritical': 0,
                                                                           'Diagnostic': 1, 'MaintenanceDesc': [],
                                                                           'MaintenanceDescDet': [],
                                                                           'TypeMaintenance': [], 'EndMaintenance': [],
                                                                           'Maintenance': 0,
                                                                           'machine_type': [], 'source': [],
                                                                           'name': [], 'alert_type': [],
                                                                           'cod': [], 'PdM': [], 'PdO': [], 'event_type': [],
                                                                           'lat': [], 'lon': []}


            else:
                if (TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance'] == 1):
                    print('Exist maintenance ', df_MaintenanceData['Inizio guasto'][ind])

            #print('Maintenance -  data store')
            TrainData[df_MaintenanceData['Inizio guasto'][ind]]['EndMaintenance'].append(df_MaintenanceData['Fine guasto'][ind])
            TrainData[df_MaintenanceData['Inizio guasto'][ind]]['TypeMaintenance'].append(df_MaintenanceData['alert_type avviso'][ind])
            TrainData[df_MaintenanceData['Inizio guasto'][ind]]['MaintenanceDesc'].append(df_MaintenanceData['Descrizione'][ind])
            TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance'] = TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance'] + 1

            # if df_MaintenanceData['Inizio guasto'][ind] not in storedate:
            #    print('Found match, ', df_MaintenanceData['Inizio guasto'][ind])
            #    storedate.append(df_MaintenanceData['Inizio guasto'][ind])

    print('Total Maintenance event (TCU+DCU) = ', totalmaintenanceevent, '      Number of maintenance event (TCU)= ', nummaintenanceevent)
    # store train data in a pickle file
    outputdataAnalysisfile = outputlocation + '\\' + trainID + '-' + outputfname
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))

    #outputdataAnalysisfile = outputlocation+'\\'+trainID+'-'+outputdiagnosticfname
    #pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))
    return TrainData

#-----------------------------------------------------------------------------------------------------------------------------
# Function : ProcessDiagnosticData
# Description : This function takes into account the diagnostic events of a train and prepare a time-series of the diagnostic events
# with important features (i.e., alerts) and store
# Input: Raw .csv file containing list of diagnostic events of a train
# Output: A dictionary where the key is the timestamp of events. Each event is associated with other important features of
# the diagnostic event such as, name, alert_type, cod, PdM, PdO, event_type, depot
#
# For the data cleaning we follow the below heuristic:
# discard data if 'depot' == 1
#-----------------------------------------------------------------------------------------------------------------------------
def ProcessDiagnosticData(df_DiagnosticData,AlertCode, trainID, trainCarList):
    TrainData={}
    ListMissingAlermCode=[]
    CountPDMwithoutPDO=0
    considercomponentcode=0
    numofTCUDiganosticeventbeforefiltering=0
    Type =""

    print('================Start reading raw diagnostic event from file---------------------------------')
    numrecord=0
   # iterate through each row of the diagnostic data
    for ind in df_DiagnosticData.index:
        numrecord=numrecord+1
        sametraindata=0
        #if df_DiagnosticData['source'][ind] in trainCarList:  # check whether all diagnostic events are from the same train
        #    sametraindata=1

    #--------------Step 1---------------------------------------------------------------------------------------------------
    #   First check whether it is a PDM or PDO, we usually go for PDO, only cosider a PDM if there is no associated PDO with it
    #-----------------------------------------------------------------------------------------------------------------------
        if(int(str(df_DiagnosticData['id1'][ind]).split()[0])==0):
            Type ='PDM'
            Alertkey = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],
                                       str(df_DiagnosticData['id'][ind]).split()[0], str('-1'))

            if Alertkey in listofPDMOnly:  # only consider the PDM that is not associated with any PDO, otherwise discard
                CountPDMwithoutPDO = CountPDMwithoutPDO + 1
            else:
                #print('PDM --discarding')
                continue
        else:
            Type = 'PDO'
            Alertkey = GetAlertCodeKey(str(df_DiagnosticData['cod'][ind]).split()[0],
                                       str(df_DiagnosticData['id'][ind]).split()[0],
                                       str(df_DiagnosticData['id1'][ind]).split()[0])

    # --------------Step 2---------------------------------------------------------------------------------------------------
    #   If we are fine with the alert code, then we go for making an entry, check if this alert is an duplicate one
    # -----------------------------------------------------------------------------------------------------------------------
        if df_DiagnosticData['ts'][ind] not in TrainData.keys():
            #newRecordinDictionary = 1  # indicates that this event is a new event with a new timestamp
            #print('newRecordinDictionary = ', newRecordinDictionary, 'It is a New entry=', df_DiagnosticData['ts'][ind])
            TrainData[df_DiagnosticData['ts'][ind]] = {'Alertkey':[], 'AlertColor':[], 'AlertLevel':[],
                                                       'CriticalAlertCode':[], 'CriticalAlertColor':[],
                                                       'NumPDO':0, 'NumPDM':0,'NumAlert':0,'NumCritical':0,
                                                       'Diagnostic': 1,'MaintenanceDesc':[],'MaintenanceDescDet':[],
                                                       'TypeMaintenance':[],'EndMaintenance':[], 'Maintenance':0,
                                                       'machine_type':[],'source':[],'name': [], 'alert_type': [],
                                                       'cod': [],'PdM': [], 'PdO': [], 'event_type': [],
                                                       'lat': [],'lon': []}



        # Check for the alertkey of the alert event into alert database
        if '0-0-0' not in Alertkey:  # duplicatealert=0 indicate no prev entry for this alert is already stored in this timestamp
            #print('False Alerm')
            if((Alertkey in AlertCode.keys())):
                TrainData[df_DiagnosticData['ts'][ind]]['Alertkey'].append(Alertkey)
                TrainData[df_DiagnosticData['ts'][ind]]['AlertColor'].append(AlertCode[Alertkey]['Colore'])
                TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel'].append(AlertCode[Alertkey]['Critico'])
                if (AlertCode[Alertkey]['Critico']==1):
                    TrainData[df_DiagnosticData['ts'][ind]]['CriticalAlertCode'].append(Alertkey)
                    TrainData[df_DiagnosticData['ts'][ind]]['CriticalAlertColor'].append(AlertCode[Alertkey]['Colore'])
            else:
                print('-----------------------Alert key not found = ', Alertkey)
                # pass
        else:
            print('False Alert key = ', Alertkey)
            pass

        #print('Alert key = ', Alertkey, 'index = ', ind)
        TrainData[df_DiagnosticData['ts'][ind]]['name'].append(df_DiagnosticData['name'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['alert_type'].append(df_DiagnosticData['alert_type'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['cod'].append(df_DiagnosticData['cod'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['PdM'].append(df_DiagnosticData['id'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['PdO'].append(df_DiagnosticData['id1'][ind])
        #Calculated value
        TrainData[df_DiagnosticData['ts'][ind]]['NumPDO']=len(TrainData[df_DiagnosticData['ts'][ind]]['PdO'])
        TrainData[df_DiagnosticData['ts'][ind]]['NumPDM']=len(TrainData[df_DiagnosticData['ts'][ind]]['PdM'])


        TrainData[df_DiagnosticData['ts'][ind]]['NumAlert']=len(TrainData[df_DiagnosticData['ts'][ind]]['Alertkey'])
        TrainData[df_DiagnosticData['ts'][ind]]['NumCritical']=(np.count_nonzero(TrainData[df_DiagnosticData['ts'][ind]]['AlertLevel']))

        TrainData[df_DiagnosticData['ts'][ind]]['event_type'].append(df_DiagnosticData['event_type'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['lat'].append(df_DiagnosticData['lat'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['lon'].append(df_DiagnosticData['lon'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['source'].append(df_DiagnosticData['source'][ind])
        TrainData[df_DiagnosticData['ts'][ind]]['machine_type'].append(df_DiagnosticData['machine_type'][ind])

    #print(len(uniqueEntry))
    #print(uniqueEntry)
    print('Diagnostic Data Summary')
    print('---------------------------------------------------------------')
    #print('Number of PDMwithoutPDO = ', CountPDMwithoutPDO)
    print('Total diagnostic event (TCU+DCU) recorded in the file = ', numrecord)
    print('Total TCU diagnostic events in the file (before filtering) = ', numofTCUDiganosticeventbeforefiltering)
    print('Number of days where diagnostic events are found = ', len(TrainData.keys()))
    #print(TrainData)

    #print('Size of Data dictionary  ', len(TrainData.keys()))
    print('Missing alerm code = ', len(ListMissingAlermCode))
    #print(ListMissingAlermCode)
    print('List of days =  ', TrainData.keys())

  #---------------------------------------------------------------------------------------------------------------------------------
  #  Step 3 : Take the half of the alerts and critical alerts to discard the duplicate PDO (as in theory each PDO appears twice in the event log)
  #----------------------------------------------------------------------------------------------------------------------------------
    totalalertfortrain=0
    totalcriticalalertfortrain=0
    for tkey in TrainData.keys():
        #print('Key = ', tkey)
        befalert = TrainData[tkey]['NumAlert']
        befcriticalalert= TrainData[tkey]['NumCritical']
        if( TrainData[tkey]['NumAlert'] >0):
            TrainData[tkey]['NumAlert'] = math.ceil(TrainData[tkey]['NumAlert'] /2)
            totalalertfortrain=totalalertfortrain+ TrainData[tkey]['NumAlert']

        if( TrainData[tkey]['NumCritical']>0):
            TrainData[tkey]['NumCritical'] = math.ceil(TrainData[tkey]['NumCritical']/2)
            totalcriticalalertfortrain=totalcriticalalertfortrain+TrainData[tkey]['NumCritical']
        #print('Key = ', tkey, '   Bef alert = ',befalert, '    Bef Num critical = ',befcriticalalert,   'Num alerts = ', TrainData[tkey]['NumAlert'], '   Critical = ', TrainData[tkey]['NumCritical'])
        #print('Key = ', tkey, '   num alert = ', TrainData[tkey]['NumAlert'], '   Num critial alert = ', TrainData[tkey]['NumCritical'])

    print('Number of alerts (after all filtering) in the file = ', totalalertfortrain)
    print('Number of critical alerts (after all filtering) in the file = ', totalcriticalalertfortrain)

    outputdataAnalysisfile = outputlocation+'\\'+trainID+'-'+outputdiagnosticfname
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))
    return TrainData
#-------------------------------------------------------------------------------------
#Function : GetAlertCodeKey
# Function to retrieve the alert key from the diagnostic data
#-------------------------------------------------------------------------------------
def GetAlertCodeKey(groupCodename, subgroup1Codename, subgroup2Codename):
    groupkey=""
    #print('in function GetAlertCodeKey() = ', groupCodename, subgroup1Codename, subgroup2Codename)
    #groupCode = groupCodename.split()
    #groupCodeN = groupCode[0]
    groupCodeNum =  int(groupCodename)
    countnanEntry=0

    if pd.isna(float(subgroup1Codename)) == False:
        #print('subgroup')
        #subgroup1Code = str(subgroup1Codename).split()
        #subgroup1CodeN = subgroup1Code[0]
        subgroup1CodeNum = int(subgroup1Codename)
    else:
        countnanEntry = countnanEntry+1
        subgroup1CodeNum=-1

    if pd.isna(float(subgroup2Codename)) == False:
        #subgroup2Code = str(subgroup2Codename).split()
        #subgroup2CodeN = subgroup2Code[0]
        subgroup2CodeNum = int(subgroup2Codename)
    else:
        countnanEntry = countnanEntry+1
        subgroup2CodeNum=-1

    #print('Group code = ', groupCodeNum, '  subgroup1 = ', subgroup1CodeNum, 'subgroup2 = ', subgroup2CodeNum, 'nan entry = ', countnanEntry)

    # make a key with the 'group_subgroup1_subgroup2' and put it into the dictionary
    if (countnanEntry<2):

        groupkey = str(groupCodeNum) + '-' + str(subgroup1CodeNum) + '-' + str(subgroup2CodeNum)
    #else:
    #    print('No code')

    #print('groupkey value = ', groupkey)
    return groupkey

#-------------------------------------------------------------------------------------
#Function : LoadAlertCode
# This function loads the list of alert codes from a file
#-------------------------------------------------------------------------------------
def LoadAlertCode(TCUalertfileLocation, datasheetname):
    AlertCode={}
    NumberOfCriticalAlerts =0

    #print('----------Loading Alert code for TCU---------: ' + TCUalertfileLocation)
    df_alertcodeTCU = pd.read_excel(TCUalertfileLocation, sheet_name=datasheetname)
    #print(df_alertcodeTCU.head(20))

    # iterating over the list of TCU alerts
    for ind in df_alertcodeTCU.index:
        #print('ind = ', ind, 'Critico : ', df_alertcodeTCU['Critico'][ind],'gcode = ', df_alertcodeTCU['Codice gruppo'][ind], '  ',df_alertcodeTCU['Codice sottogruppo  1'][ind], '  ', df_alertcodeTCU['Codice sottogruppo  2'][ind])
        groupkey = GetAlertCodeKey(str(df_alertcodeTCU['Codice gruppo'][ind]).split()[0], str(df_alertcodeTCU['Codice sottogruppo  1'][ind]).split()[0],str(df_alertcodeTCU['Codice sottogruppo  2'][ind]).split()[0])

        if(groupkey!=""):
            if groupkey in AlertCode.keys():
                pass
                #print('key exist =', groupkey)
            else:
                #print('New entry = ', groupkey)
                AlertCode[groupkey] = {'Critico':0, 'Colore': "", 'Descrizione lunga': "", 'Guida Operatore': ""}
                AlertCode[groupkey]['Colore'] = df_alertcodeTCU['Colore'][ind]
                if(df_alertcodeTCU['Critico'][ind]=='X'):
                    AlertCode[groupkey]['Critico'] = 1
                    NumberOfCriticalAlerts=NumberOfCriticalAlerts+1
                #AlertCode[groupkey]['Colore'] = df_alertcodeTCU['Colore'][ind]
                AlertCode[groupkey]['Descrizione lunga'] = df_alertcodeTCU['Descrizione lunga'][ind]
                #AlertCode[groupkey]['Guida Operatore'] = df_alertcodeTCU['Guida Operatore'][ind]
    return AlertCode

#-------------------------------------------------------------------------------------
#Function : LoadDiagnosticData
#Description : This function loads the raw diagnostic data of a train and fileter out some irrlevant rows and store the data in a panda dataframe
#Input : Location and name of the file containing diagnostic data of a train
#Output : A panda dataframe containing the diagnostic data
#-------------------------------------------------------------------------------------
def LoadDiagnosticData(trainDiagnosticfilename):
    print('----------Loading Diagnostic Data--------: '+trainDiagnosticfilename)
    col_names = ['source', 'name', 'machine_type',
             'alert_type', 'ts', 'cod',
             'id', 'id1', 'event_type',
             'master', 'pantograph', 'speed',
             'lat', 'lon', 'extra', 'depot']

    df_DiagnosticData = pd.read_csv(trainDiagnosticfilename, usecols=range(0, 16), low_memory=False)

    #Look for the specific components data
    if (trainComp =='TCU'):
        discardcomponents=7  # 5 indicate TCU unit, 7 indicates DCU unit
        print('Loading TCU data...')

    if (trainComp =='DCU'):
        discardcomponents=5  # 5 indicate TCU unit, 7 indicates DCU unit
        print('Loading DCU data...')

    # Filter 1 : drop diagnostic data generated in the maintenance depot ('cod'=='TrainSet')
    df_DiagnosticData.drop(df_DiagnosticData[df_DiagnosticData['alert_type'] == 'TrainSet'].index, inplace=True)


    # Filter 2 : drop diagnostic data from other irrelevant units (for example, if consider TCU then drop data from DCU unit)
    df_DiagnosticData.drop(df_DiagnosticData[df_DiagnosticData['cod'] == discardcomponents].index, inplace=True)

    # Filter 3 : drop rows with 'depot' value 1, indicating diagnostic data generated within the maintenance systems
    df_DiagnosticData.drop(df_DiagnosticData[df_DiagnosticData['depot'] == 1].index, inplace=True)

    # Filter 4 : drop rows with 'event' value 0
    df_DiagnosticData.drop(df_DiagnosticData[df_DiagnosticData['event_type'] == 0].index, inplace=True)

    #Converting the Unix timestamp into datetime object
    df_DiagnosticData['ts'] = pd.to_datetime(df_DiagnosticData['ts'], unit='ms').dt.date
    #print(df_DiagnosticData['ts'].head(30))
    #print(df_DiagnosticData['cod'].head(30))
    return df_DiagnosticData

#-------------------------------------------------------------------------------------
#Function : LoadMaintenanceData
#-------------------------------------------------------------------------------------
def LoadMaintenanceData(trainMaintenanceDatafilename, datasheetname):
    #print('----------Loading Maintenance Data--------Train: ' + datasheetname)
    df_MaintenanceData = pd.read_excel(trainMaintenanceDatafilename, sheet_name=datasheetname)
    df_MaintenanceData['Inizio guasto'] = pd.to_datetime(df_MaintenanceData['Inizio guasto']).dt.date
    #print(df_MaintenanceData['Inizio guasto'].tail())
    #print(df_MaintenanceData['Inizio guasto'].head(20))
    return df_MaintenanceData


#---------------------------------------------------------------------------------------------------------
# main() functioN -
# Input parameter - (a) Name and number of the train
#----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    trainname ='TSR'
    trainnumber = '017'  # T1 =070, T2 = 086, T3 =040, T4 =008

    # -------------------------------------------------------------------------------
    # List of train cars - currently hard coded
    # For train T1 = number 070 ['711-162','710-183','710-184','710-185','710-161']
    # For train T2= number 086 ['711-153', '710-151', '710-153', '710-154', '710-152', '711-156']
    # For train T3 = number 040 ['711-083','710-135','710-131','710-132','710-170','711-084']
    # For train T4 = number 008 ['711-063','711-035','711-062']
    #----------------------------------------------------------------------------------
    trainCarList =[]
    if(trainnumber=='070'):
        trainCarList =['711-162','710-183','710-184','710-185','710-161']
        print('Train Number = ', trainnumber, '   List of Cars= ', trainCarList)
    if (trainnumber == '086'):
        trainCarList = ['711-153', '710-151', '710-153', '710-154', '710-152', '711-156']
        print('Train Number = ', trainnumber, '   List of Cars= ', trainCarList)
    if (trainnumber == '040'):
        trainCarList = ['711-083','710-135','710-131','710-132','710-170','711-084']
        print('Train Number = ', trainnumber, '   List of Cars= ', trainCarList)
    if (trainnumber == '008'):
        trainCarList = ['711-063','711-035','711-062']
        print('Train Number = ', trainnumber, '   List of Cars= ', trainCarList)
    #----------------------------------------------------------------------------------

    #traincomponent ='TCU'
    trainDiagnosticfilename =dataInputLocation +"\\DiagnosticData\\"+trainname+" "+trainnumber+".csv"
    trainMaintenanceDatafilename = dataInputLocation +"\\MaintenanceData\\Avvisi SAP - Interventi manutentivi.xlsx"
    datasheetname =trainname+" "+trainnumber
    trainID = trainname+trainnumber

    #Load list of alert code in a dictionary
    AlertCode={}
    TCUalertfileLocation = dataInputLocation +"\\Codici diagnostici TCU criticiFinal.xlsx" #Codici diagnostici TCU critici.xlsx"
    #alertfilelocation = dataInputLocation +"\\Codici diagnostici TCU - DCU.xlsx"
    AlertCode = LoadAlertCode(TCUalertfileLocation, "Allarmi critici")

    df_DiagnosticData = LoadDiagnosticData(trainDiagnosticfilename)  #Load diagnostic Data
    #df_MaintenanceData = LoadMaintenanceData(trainMaintenanceDatafilename, datasheetname)  #Load maintenance Data

    #Processing raw data
    TrainDataDiagnosticMaintenance={}
    TrainDiagnosticData={}
    TrainDiagnosticData = ProcessDiagnosticData(df_DiagnosticData, AlertCode, trainID, trainCarList)  # Process only the diagnostic data
    #TrainDataDiagnosticMaintenance = ProcessDiagnosticMaintenanceData(df_DiagnosticData,df_MaintenanceData,AlertCode, trainID, trainCarList) # Process both diagnostic and maintenance data
