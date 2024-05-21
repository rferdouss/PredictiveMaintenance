#======================================================================================================================
# DataProcessingTrenordDiagnosticMaintenanceRecords.py -
# Description :  This python script is to process and clean the diagnostic and maintenance data of Trenord (From July 2020 - March 2023)

# Input : A .csv file containing diagnostic record of a train
# Output: Python dictionary

# Raihana Ferdous
# Date : 16/05/2024
#==========================================================================================================================
import math
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd


#---------------------------Global variables---------------------------------------------------------------------------------
dataInputLocation = "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\DataInput"
outputlocation= "C:\\Raihana-Work-CNR\\Work-CNR-From-October2023\\Project\\PNRR-Ferroviario\\WorkingDrive\\DataFromTrenord\\Analysis-Raihana\\Output"
trainComp = 'TCU'   # considered train component
outputfname=trainComp+"-DiagnosticMaintenanceData"
outputdiagnosticfname=trainComp+"-DiagnosticData"

listofPDMOnly=['5-17--1','5-8--1', '5-6--1','5-4--1', '5-2--1']  # For TCU - List of PDM alert that are not associated with any PDO
#------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
# Function : ProcessDiagnosticMaintenanceData
# This function is used to process both diagnostic and maintenance data and put them in a python dictionary
# Important features of DiagnosticData : Vettura, Tipo, Gruppo, PdM, PdO, Evento, Deposito
# discard data if 'Deposito' == 1
#-----------------------------------------------------------------------------------------------------------------------------
def ProcessDiagnosticMaintenanceData(df_DiagnosticData,df_MaintenanceData, AlertCode, trainID):
    TrainData={}
    ListMissingAlermCode=[]

    considercomponentcode=0
    if (trainComp =='TCU'):
        considercomponentcode=5
        print('Considering TCU data')

    if (trainComp =='DCU'):
        considercomponentcode=7
        print('Considering DCU data')


    # iterate through each row of the diagnostic data
    for ind in df_DiagnosticData.index:
        if df_DiagnosticData['Gruppo'][ind]==considercomponentcode:  # consider TCU records only
            #print(' Data group = ',  df_DiagnosticData['Gruppo'][ind])
            if df_DiagnosticData['Deposito'][ind]!=1 and df_DiagnosticData['Evento'][ind]=='ON':   # discard rows with df['Deposito'][ind]==1, as they are not relevant, reference dataprocessing notes from Trenord
                print('deposit = ', df_DiagnosticData['Deposito'][ind], '  event = ', df_DiagnosticData['Evento'][ind])
                #print(df_DiagnosticData['Unix_timestamp(ms)'][ind], df_DiagnosticData['Vettura'][ind])
                if df_DiagnosticData['Unix_timestamp(ms)'][ind] not in TrainData.keys():
                    #print('New entry=', df_DiagnosticData['Unix_timestamp(ms)'][ind])
                    TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]] = {'Alertkey':[], 'AlertColor':[], 'AlertLevel':[],
                                                                               'NumPDO':0, 'NumPDM':0,'NumAlert':0,'NumCritical':0,
                                                                               'Diagnostic': 1,'MaintenanceDesc':[],'MaintenanceDescDet':[],'TypeMaintenance':[],'EndMaintenance':[], 'Maintenance':0,'Monitor':[],'Vettura': [], 'Tipo': [], 'Gruppo': [],
                                                                               'PdM': [], 'PdO': [], 'Evento': [], 'Latitudine': [], 'Longitudine': []}


                #print('pdm = ', str(df_DiagnosticData['PdM'][ind]).split()[0],'   pdo = ',   str(df_DiagnosticData['PdO'][ind]))
                if(df_DiagnosticData['Tipo'][ind]=='PDM'):
                    Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0], str(df_DiagnosticData['PdM'][ind]).split()[0], str('-1'))
                    #print('it is PDM =  ', Alertkey)
                else:
                    Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(df_DiagnosticData['PdO'][ind]).split()[0])
                    #print('PDO=  ', Alertkey)

                #Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(df_DiagnosticData['PdO'][ind]).split()[0])
                print('Data type = ', df_DiagnosticData['Tipo'][ind], '  index = ', ind, '  Alert key in diagnostic data = ', Alertkey)

                if '0-0-0' not in Alertkey:
                    #print('False Alerm')
                    if((Alertkey in AlertCode.keys())):
                        #print('AlertKey found')
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'].append(Alertkey)
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertColor'].append(AlertCode[Alertkey]['Colore'])
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'].append(AlertCode[Alertkey]['Critico'])
                    else:
                        print('-----------------------Alert key not found = ', Alertkey)
                        if (str(df_DiagnosticData['PdO'][ind]) == '0'):
                            AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(float("nan")))
                        else:
                            if(pd.isna(float(str(df_DiagnosticData['PdO'][ind]))) == True):
                                AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],
                                                                str(df_DiagnosticData['PdM'][ind]).split()[0],
                                                                str('0'))
                            #print('Alternate alert key = ', AlertkeyAlternate)
                            if(AlertkeyAlternate in AlertCode.keys()):
                                #print('Alternate alert key found in Alerm list = ', AlertkeyAlternate)
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'].append(AlertkeyAlternate)
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertColor'].append(AlertCode[AlertkeyAlternate]['Colore'])
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'].append(AlertCode[AlertkeyAlternate]['Critico'])
                            else:
                                print('Finally Alternate alerm code is not found = ', AlertkeyAlternate, 'orignal key=  ',Alertkey)
                                ListMissingAlermCode.append(AlertkeyAlternate)
                else:
                    #print('False Alerm = ', Alertkey)
                    pass


                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Vettura'].append(df_DiagnosticData['Vettura'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Tipo'].append(df_DiagnosticData['Tipo'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Gruppo'].append(df_DiagnosticData['Gruppo'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdM'].append(df_DiagnosticData['PdM'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdO'].append(df_DiagnosticData['PdO'][ind])
                #print('To check = ', (np.count_nonzero(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'])), "      = ", TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'])
                #Calculated value
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumPDO']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdO'])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumPDM']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdM'])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumAlert']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumCritical']=(np.count_nonzero(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel']))

                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Evento'].append(df_DiagnosticData['Evento'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Latitudine'].append(df_DiagnosticData['Latitudine'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Longitudine'].append(df_DiagnosticData['Longitudine'][ind])
                #TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Sorgente_Dati'].append(df_DiagnosticData['Sorgente_Dati'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Monitor'].append(df_DiagnosticData['Monitor'][ind])
    #print(len(uniqueEntry))
    #print(uniqueEntry)
    print('Number of entry in the train data', len(TrainData.keys()))
    #print(TrainData)

    #Maintenance Data : iterate through each row of the maintenance data
    storedate=[]
    for ind in df_MaintenanceData.index:
        print('maintenance date = ', df_MaintenanceData['Inizio guasto'][ind])
        if df_MaintenanceData['Inizio guasto'][ind] not in TrainData.keys(): # if already an entry exists for keeping the diagnostic data for this day
            TrainData[df_MaintenanceData['Inizio guasto'][ind]] = {'Alertkey':[], 'AlertColor':[], 'AlertLevel':[],
                                                                   'NumPDO':0, 'NumPDM':0,'NumAlert':0,'NumCritical':0,
                                                                   'Diagnostic': 0, 'MaintenanceDesc': [], 'MaintenanceDescDet': [],
                                                                       'TypeMaintenance': [], 'EndMaintenance': [],
                                                                       'Maintenance': 0, 'Monitor': [], 'Vettura': [],
                                                                       'Tipo': [], 'Gruppo': [],
                                                                       'PdM': [], 'PdO': [], 'Evento': [],
                                                                       'Latitudine': [], 'Longitudine': []}

        else:
            if (TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance']==1):
                print('Exist maintenance ', df_MaintenanceData['Inizio guasto'][ind])
        TrainData[df_MaintenanceData['Inizio guasto'][ind]]['EndMaintenance'].append(df_MaintenanceData['Fine guasto'][ind])
        TrainData[df_MaintenanceData['Inizio guasto'][ind]]['TypeMaintenance'].append(df_MaintenanceData['Tipo avviso'][ind])
        TrainData[df_MaintenanceData['Inizio guasto'][ind]]['MaintenanceDesc'].append(df_MaintenanceData['Descrizione'][ind])
        TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance']=TrainData[df_MaintenanceData['Inizio guasto'][ind]]['Maintenance']+1
        #TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Tipo'].append(df_DiagnosticData['Tipo'][ind])
        #TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Gruppo'].append(df_DiagnosticData['Gruppo'][ind])
        #TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdM'].append(df_DiagnosticData['PdM'][ind])
        #TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdO'].append(df_DiagnosticData['PdO'][ind])

            #if df_MaintenanceData['Inizio guasto'][ind] not in storedate:
            #    print('Found match, ', df_MaintenanceData['Inizio guasto'][ind])
            #    storedate.append(df_MaintenanceData['Inizio guasto'][ind])
    print('Size of Data dictionary  ', len(TrainData.keys()))
    print('Missing alerm code = ', len(ListMissingAlermCode))
    print(ListMissingAlermCode)
    #print(TrainData)
    #store train data in a pickle file
    outputdataAnalysisfile = outputlocation+'\\'+trainID+'-'+outputfname
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))


    return TrainData

#----------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------
def CheckForDuplicateAlertKey(TrainData, timekey, tid, caid, alertcode, monitorid):
    duplicatealert=0
    print('In funciton checkforduplicatealertkey,    alertkey = ', alertcode, '   monitor id = ', monitorid)
    print('Car info = ', len(TrainData[timekey]['Vettura']))
    print('Train source = ', len(TrainData[timekey]['Sorgente_Dati']))
    print('Monitor = ', len(TrainData[timekey]['Monitor']))
    print('Alertkey = ', len(TrainData[timekey]['Alertkey']))

    # check for duplicate entry
    i=0
    for i in range(len(TrainData[timekey]['Vettura'])):
        trainid = TrainData[timekey]['Sorgente_Dati'][i]
        carid = TrainData[timekey]['Vettura'][i]
        monid = TrainData[timekey]['Monitor'][i]
        alertid = TrainData[timekey]['Alertkey'][i]
        if(trainid == tid and carid==caid and alertid == alertcode and monid==monitorid):
            duplicatealert=1
            #print('trainid = ', trainid, tid, '   carid = ', carid, caid, ' monitor id = ', monid, monitorid, '   alert key = ', alertid, alertcode)
        #else:
        #    print('Not duplicate')
            #print('trainid = ', trainid, tid, '   carid = ', carid, caid, ' monitor id = ', monid, monitorid,'   alert key = ', alertid, alertcode)

    return duplicatealert

#-----------------------------------------------------------------------------------------------------------------------------
# Function : ProcessDiagnosticData
# Description : This function takes into account the diagnostic events of a train and prepare a time-series of the diagnostic events
# with important features (i.e., alerts) and store
# Input: Raw .csv file containing list of diagnostic events of a train
# Output: A dictionary where the key is the timestamp of events. Each event is associated with other important features of
# the diagnostic event such as, Vettura, Tipo, Gruppo, PdM, PdO, Evento, Deposito
#
# For the data cleaning we follow the below heuristic:
# discard data if 'Deposito' == 1
#-----------------------------------------------------------------------------------------------------------------------------
def ProcessDiagnosticData(df_DiagnosticData,AlertCode, trainID, trainCarList):
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

        if df_DiagnosticData['Sorgente_Dati'][ind] in trainCarList:  # check whether all diagnostic events are from the same train
            sametraindata=1

        if df_DiagnosticData['Gruppo'][ind]==considercomponentcode and sametraindata==1:  # consider TCU records only
            numofTCUDiganosticeventbeforefiltering= numofTCUDiganosticeventbeforefiltering+1
            #print(' Data group = ',  df_DiagnosticData['Gruppo'][ind])
            #print('---Event Index = ', ind,  '   type = ', df_DiagnosticData['Tipo'][ind], 'deposit = ', df_DiagnosticData['Deposito'][ind], '  event = ', df_DiagnosticData['Evento'][ind])
            if df_DiagnosticData['Deposito'][ind]!=1 and df_DiagnosticData['Evento'][ind]=='ON':   # discard rows with df['Deposito'][ind]==1, as they are not relevant, reference dataprocessing notes from Trenord
                #print('deposit = ', df_DiagnosticData['Deposito'][ind], '  event = ', df_DiagnosticData['Evento'][ind])
                #print('Time of event = ', df_DiagnosticData['Unix_timestamp(ms)'][ind], '   Car Num= ', df_DiagnosticData['Vettura'][ind])
                # print('pdm = ', str(df_DiagnosticData['PdM'][ind]).split()[0],'   pdo = ',   str(df_DiagnosticData['PdO'][ind]))


    #--------------Step 1---------------------------------------------------------------------------------------------------
    #   First check whether it is a PDM or PDO, we usually go for PDO, only cosider a PDM if there is no associated PDO with it
    #-----------------------------------------------------------------------------------------------------------------------
                #print('Group = ', str(df_DiagnosticData['Gruppo'][ind]).split()[0], '   PDM = ',str(df_DiagnosticData['PdM'][ind]).split()[0], '   PDO =', str(df_DiagnosticData['PdO'][ind]).split()[0])
                # if it is of type other than PDO and PDM (e.g., trainset) then discard
                if (df_DiagnosticData['Tipo'][ind] =='TrainSet'):
                    print('Trainset - maintenance event - continue')
                    continue

                else:  # go forward if it is a PDO or PDM
                    if(int(str(df_DiagnosticData['PdO'][ind]).split()[0])==0):
                        Type ='PDM'
                        Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdM'][ind]).split()[0], str('-1'))

                        if Alertkey in listofPDMOnly:  # only consider the PDM that is not associated with any PDO, otherwise discard
                            # print('PDM only = ', Alertkey, '    listpdmonly = ', listofPDMOnly)
                            CountPDMwithoutPDO = CountPDMwithoutPDO + 1
                        else:
                            #print('PDM --discarding')
                            continue
                    else:
                        Type = 'PDO'
                        Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdM'][ind]).split()[0],
                                                   str(df_DiagnosticData['PdO'][ind]).split()[0])
                        # print('PDO=  ', Alertkey)


                #print('Type original = ', df_DiagnosticData['Tipo'][ind],'   Type after investigation = ', Type, '  alertkey = ', Alertkey)


    # --------------Step 2---------------------------------------------------------------------------------------------------
    #   If we are fine with the alert code, then we go for making an entry, check if this alert is an duplicate one
    # -----------------------------------------------------------------------------------------------------------------------
                if df_DiagnosticData['Unix_timestamp(ms)'][ind] not in TrainData.keys():
                    newRecordinDictionary = 1  # indicates that this event is a new event with a new timestamp
                    #print('newRecordinDictionary = ', newRecordinDictionary, 'It is a New entry=', df_DiagnosticData['Unix_timestamp(ms)'][ind])
                    TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]] = {'Alertkey':[], 'AlertColor':[], 'AlertLevel':[],
                                                                               'CriticalAlertCode':[], 'CriticalAlertColor':[],
                                                                               'NumPDO':0, 'NumPDM':0,'NumAlert':0,'NumCritical':0,
                                                                               'Diagnostic': 1,'MaintenanceDesc':[],'MaintenanceDescDet':[],
                                                                               'TypeMaintenance':[],'EndMaintenance':[], 'Maintenance':0,
                                                                               'Monitor':[],'Sorgente_Dati':[],'Vettura': [], 'Tipo': [],
                                                                                'Gruppo': [],'PdM': [], 'PdO': [], 'Evento': [],
                                                                                'Latitudine': [],'Longitudine': []}



                #Alertkey = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(df_DiagnosticData['PdO'][ind]).split()[0])
                #print('Date : ', df_DiagnosticData['Unix_timestamp(ms)'][ind], '   Alert type = ', df_DiagnosticData['Tipo'][ind], '  index = ', ind, '  Alert key in diagnostic data = ', Alertkey)

                # if it is not a new entry then check if this is a duplicate alert, if so then discard otherwise go forward for storing it into the dictionary
                #if(newRecordinDictionary == 0):  # if an entry already exists
                #    print('An entry already exists for this timestamp = ', df_DiagnosticData['Unix_timestamp(ms)'][ind])
                #    duplicatealert = CheckForDuplicateAlertKey(TrainData, df_DiagnosticData['Unix_timestamp(ms)'][ind], df_DiagnosticData['Sorgente_Dati'][ind], df_DiagnosticData['Vettura'][ind], Alertkey, df_DiagnosticData['Monitor'][ind])

                # Check for the alertkey of the alert event into alert database
                if '0-0-0' not in Alertkey and duplicatealert==0:  # duplicatealert=0 indicate no prev entry for this alert is already stored in this timestamp
                    #print('False Alerm')
                    if((Alertkey in AlertCode.keys())):
                        #print('AlertKey found')
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'].append(Alertkey)
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertColor'].append(AlertCode[Alertkey]['Colore'])
                        TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'].append(AlertCode[Alertkey]['Critico'])
                        if (AlertCode[Alertkey]['Critico']==1):
                            TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['CriticalAlertCode'].append(Alertkey)
                            TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['CriticalAlertColor'].append(AlertCode[Alertkey]['Colore'])

                    else:
                        print('-----------------------Alert key not found = ', Alertkey)
                        if (str(df_DiagnosticData['PdO'][ind]) == '0'):
                            AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],str(df_DiagnosticData['PdM'][ind]).split()[0],str(float("nan")))
                        else:
                            if(pd.isna(float(str(df_DiagnosticData['PdO'][ind]))) == True):
                                AlertkeyAlternate = GetAlertCodeKey(str(df_DiagnosticData['Gruppo'][ind]).split()[0],
                                                                str(df_DiagnosticData['PdM'][ind]).split()[0],
                                                                str('0'))
                            #print('Alternate alert key = ', AlertkeyAlternate)
                            if(AlertkeyAlternate in AlertCode.keys()):
                                #print('Alternate alert key found in Alerm list = ', AlertkeyAlternate)
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'].append(AlertkeyAlternate)
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertColor'].append(AlertCode[AlertkeyAlternate]['Colore'])
                                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'].append(AlertCode[AlertkeyAlternate]['Critico'])
                            else:
                                print('Finally Alternate alerm code is not found = ', AlertkeyAlternate, 'orignal key=  ',Alertkey)
                                ListMissingAlermCode.append(AlertkeyAlternate)
                else:
                    print('False Alerm = ', Alertkey)
                    pass


                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Vettura'].append(df_DiagnosticData['Vettura'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Tipo'].append(df_DiagnosticData['Tipo'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Gruppo'].append(df_DiagnosticData['Gruppo'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdM'].append(df_DiagnosticData['PdM'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdO'].append(df_DiagnosticData['PdO'][ind])
                #print('To check = ', (np.count_nonzero(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'])), "      = ", TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel'])
                #Calculated value
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumPDO']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdO'])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumPDM']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['PdM'])


                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumAlert']=len(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Alertkey'])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['NumCritical']=(np.count_nonzero(TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['AlertLevel']))

                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Evento'].append(df_DiagnosticData['Evento'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Latitudine'].append(df_DiagnosticData['Latitudine'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Longitudine'].append(df_DiagnosticData['Longitudine'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Sorgente_Dati'].append(df_DiagnosticData['Sorgente_Dati'][ind])
                TrainData[df_DiagnosticData['Unix_timestamp(ms)'][ind]]['Monitor'].append(df_DiagnosticData['Monitor'][ind])

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

    outputdataAnalysisfile = outputlocation+'\\'+trainID+'-'+outputdiagnosticfname
    pickle.dump(TrainData, open(outputdataAnalysisfile, "wb"))


    return TrainData


#-------------------------------------------------------------------------------------
#Function : LoadDiagnosticData
#This function loads and process raw diagnostic data and store it into a dictionary
#-------------------------------------------------------------------------------------
def LoadDiagnosticData(trainDiagnosticfilename):
    #print('----------Loading Diagnostic Data--------: '+trainDiagnosticfilename)
    df_DiagnosticData = pd.read_csv(trainDiagnosticfilename, low_memory=False)
    #df_DiagnosticData['Unix_timestamp(ms)'] = pd.to_datetime(df_DiagnosticData['Unix_timestamp(ms)'], unit='ms').dt.normalize()
    df_DiagnosticData['Unix_timestamp(ms)'] = pd.to_datetime(df_DiagnosticData['Unix_timestamp(ms)'],unit='ms').dt.date

    #print(df_DiagnosticData['Unix_timestamp(ms)'].head(30))
    #print(df_DiagnosticData.tail())
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
        #else:
        #    print('group key is not valid = ', groupkey)


    #print('Total number of alert = ', len(AlertCode.keys()), '   Total Critical Alerts =', NumberOfCriticalAlerts)
    #for akey in AlertCode.keys():
    #    print(akey, '  color: ', AlertCode[akey]['Colore'], '    criticallevel: ', AlertCode[akey]['Critico'])
    #print(AlertCode.keys())
    #print('-----------------------------------------------------------------------')
    return AlertCode


#----------------------------------------------------------------------------------------------------------
#method to load data analysis result stored in a pickle file
def LoadProcessedDiagnosticMaintenanceTrainData(filename):
    TrainDataD={}
    TrainDataD = pickle.load(open(filename, "rb"))
    #print(TrainDataD)
    return TrainDataD

#------------------------------------------------------------------------------------

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    processdataflag= True # True = perform data processing, False= load processed data and draw plots

    trainname ='TSR'
    trainnumber = '086'  # T1 =070, T2 = 086, T3 =040, T4 =008

    # -------------------------------------------------------------------------------
    # List of train cars -
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
    traincomponent ='TCU'
    trainDiagnosticfilename =dataInputLocation +"\\DiagnosticData\\"+trainname+" "+trainnumber+".csv"
    trainMaintenanceDatafilename = dataInputLocation +"\\MaintenanceData\\Avvisi SAP - Interventi manutentivi.xlsx"
    datasheetname =trainname+" "+trainnumber
    trainID = trainname+trainnumber

    #Load list of alert code in a dictionary
    AlertCode={}
    TCUalertfileLocation = dataInputLocation +"\\Codici diagnostici TCU criticiEditedPDO.xlsx" #Codici diagnostici TCU critici.xlsx"
    #alertfilelocation = dataInputLocation +"\\Codici diagnostici TCU - DCU.xlsx"
    AlertCode = LoadAlertCode(TCUalertfileLocation, "Foglio1")

    if(processdataflag==True):  #Load raw diagnostic & maintenance Data and analyze it
        df_DiagnosticData = LoadDiagnosticData(trainDiagnosticfilename)  #Load diagnostic Data
        df_MaintenanceData = LoadMaintenanceData(trainMaintenanceDatafilename, datasheetname)  #Load maintenance Data

        #Processing data
        TrainDataDiagnosticMaintenance={}
        TrainDiagnosticData={}
        #TrainDataDiagnosticMaintenance = ProcessDiagnosticMaintenanceData(df_DiagnosticData,df_MaintenanceData,AlertCode, trainID)
        TrainDiagnosticData = ProcessDiagnosticData(df_DiagnosticData,AlertCode, trainID,trainCarList)

    else: # load processed data and perform time series analysis and forecasting
        #----------------plot all train data (visualization)-------------------------------------------------------
        #CriticalAlertAllTrain() #Plot with all the train info, correct one
        #Plotalltraindata()
        #----------------------------------------------------------------------------

        print('Loading processed data file for time-series analysis : ') #outputlocation + '\\' + trainID + '-' + outputfname
        #outputpicklefilename = outputlocation + '\\' + trainID + '-' + outputfname # both diagnostic and maintenance data
        outputpicklefilename= outputlocation + '\\'+ trainID + '-'+outputdiagnosticfname # only diagnostic data

        print(outputpicklefilename)
        TrainDataDiagosticMaintenance = LoadProcessedDiagnosticMaintenanceTrainData(outputpicklefilename)
        #print(TrainDataDiagosticMaintenance)
        #DrawMaintenancemonthlyPlot(TrainDataDiagosticMaintenance, trainID)

        # Sampling the data
        intervaltime = 7 # 15 days
        TrainDataRecord = GetAlertforADuration(TrainDataDiagosticMaintenance, trainID, intervaltime)
        print('Number of days in diagnostic file = ', len(TrainDataDiagosticMaintenance.keys()), '  Bin/sampling size (days)= ', intervaltime)
        #print(TrainDataRecord)
        #DrawAlertCodeColorForSpecificInterval(TrainDataRecord)

        #---------------------------------------Preparing different monthly plots (visualization)----------------------------
        outputfilelocation = outputlocation + '\\TSR'+trainnumber+'-TCU-DiagnosticData'
        #print('Monthly critical alert coMonltyBoxPlotAlertCriticalSingleTrainde and color distribution for single train------------------------')
        #MonltyBoxPlotAlertCriticalSingleTrain(outputfilelocation)  # this one is correct
        #MonltyPlotCriticalAlertCodeColorSingleTrain(outputfilelocation)
        # --------------------------------------------------------------------------------------------------------

        #-------------Time Series Data processing for building various forecasting models------------------------------------
        #dfTrainData = pd.DataFrame.from_dict(TrainDataRecord, orient='index')
        #subdfTrainData = dfTrainData[['NumAlerts', 'NumCriticalAlerts']]  # Create new pandas DataFrame
        #varname = 'NumCriticalAlerts'  #'NumAlerts', 'NumCriticalAlerts'
        #print(subdfTrainData.head(10))

        # -------------Various Time Series Data processing for understanding underlying pattern for making decisions in forecasting----------------------------------
        #autocorrelation_plot(subdfTrainData[varname]) # #AutocorrelationCheck()  # check for autocorrelation
        #plt.show()
        #LaggingPlot(subdfTrainData,varname)
        #autocorrelation_lag3 = subdfTrainData[varname].autocorr(lag=3)
        #autocorrelation_lag5 = subdfTrainData[varname].autocorr(lag=5)
        #autocorrelation_lag10 = subdfTrainData[varname].autocorr(lag=10)
        #autocorrelation_lag15 = subdfTrainData[varname].autocorr(lag=15)
        #autocorrelation_lag20 = subdfTrainData[varname].autocorr(lag=20)
        #print("Three Month Lag: ", autocorrelation_lag3,autocorrelation_lag5,autocorrelation_lag10,autocorrelation_lag15,autocorrelation_lag20)


        #DecompositionOfTimeSeriesData(subdfTrainData, varname, intervaltime)  # decomposition of time series into trend and seasonality
        #RollingStatistics(subdfTrainData, varname)
        #DrawACFandPACFPlot(subdfTrainData, varname)
        #CheckStationarityInTimeSeriesData(subdfTrainData, varname, 12)  # check for stationary in data
        #exit(0)

        # ----------Experiments with different Time series forecasting models------------------------------------------------------------------------
        #traindatapercentage= 0.70

        #NaiveRMSE = NaiveForecastingModel(subdfTrainData, varname, traindatapercentage)


        #windowsize = 10  # here 10 weeks
        #MA_RMSE = MovingAverage_Model(subdfTrainData, varname, traindatapercentage, windowsize)

        #arima_RMSE = ARIMA_Model(subdfTrainData, varname, traindatapercentage)#(subdfTrainData, varname, traindatapercentage)
        #sarima_RMSE = SARIMA_Model(subdfTrainData, varname, traindatapercentage)
        #tripleexponentialSmoothing_RMSE = TripleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage)
        #doubleexponentialSmoothing_RMSE = DoubleExponentialSmoothing_forecastingModel(subdfTrainData, varname, traindatapercentage)

        #ARMA_RMSE=0
        #ARMA_RMSE = ARMA_Model(subdfTrainData, varname, traindatapercentage)

        #print('Root Mean Square Error, ARIMA = ', arima_RMSE, '   SARIMA = ', sarima_RMSE)
        #print('BaseLine Naive = ', NaiveRMSE)
        #print('Moving Average (MA) = ', MA_RMSE)
        #print('ARMA Model = ', ARMA_RMSE)
        #print('ARIMA = ', arima_RMSE)
        #print('SARIMA = ', sarima_RMSE)
        #print('Double Exponential Smoothing = ', doubleexponentialSmoothing_RMSE)
        #print('Triple Exponential Smoothing = ', tripleexponentialSmoothing_RMSE)
        #exit(0)


    # ----------------------------------------------------------------------------------
        #PlotDiagnosticYearly(subdfTrainData, varname)

        # Time series analysis - Decomposition
        #DecompositionOfTimeSeriesData(subdfTrainData, varname, intervaltime) # decomposition of time-series data

        #Time Series Forecasting process  (1) check for stationarity,
        #CheckStationarityInTimeSeriesData(subdfTrainData, varname, 5) # check for stationary in data




        #ARIMA modeling steps
        #DrawACFandPACFPlot(subdfTrainData, varname)  #ARIMA- step 1: Model Definition -  specifying the p, d, and q parameters
        ##ARIMA_model = ARIMA_ModelFitting(subdfTrainData, varname) #ARIMA- step 2 ,3: Model fitting  -  Train the model. Model diagnostic
        #ARIMA_forecast(subdfTrainData, varname, ARIMA_model, periods=24) # ARIMA- step 4: Model forecasting
        #ARIMA_model.fit_predict(dynamic= False)
        #model = ARIMA(subdfTrainData[varname], order=(2, 0, 0))
        #model_fit = model.fit()
        #model_fit = ARIMA_model.fit(subdfTrainData[varname])

        ##model = sm.tsa.arima.ARIMA(subdfTrainData[varname], order=(2, 0, 0))
        ##model_fit = model.fit()
        #print(fitted.summary())


        #print(model_fit.summary())
        ##forecast = model_fit.forecast(steps=30)
        ##print(forecast)
        #----TrainingNForecast(subdfTrainData, varname)
        #---TrainPredict(subdfTrainData, varname)
        #SARIMA_Model(subdfTrainData, varname)
        #plt.show()
        #exit(0)
        #DrawBarPlot(TrainDataDiagosticMaintenance) #Drawing the plots
        #DrawMaintenancePlot(TrainDataDiagosticMaintenance, trainID)
        #DrawMaintenancemonthlyPlot(TrainDataDiagosticMaintenance, trainID)
