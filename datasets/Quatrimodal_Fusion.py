import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

ETHNICITY = {'WHITE': 0,
 'UNKNOWN': 1,
 'OTHER': 2,
 'BLACK/AFRICAN AMERICAN': 3,
 'HISPANIC/LATINO': 4,
 'ASIAN': 5,
 'AMERICAN INDIAN/ALASKA NATIVE': 6,
 'UNABLE TO OBTAIN': 7}

GENDER = {'M': 0, 'F': 1}


R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]
    
# Define age buckets
def age_to_bucket(age):
    if age <= 20:
        return 0
    elif age <= 40:
        return 1
    elif age <= 60:
        return 2
    elif age <= 80:
        return 3
    else:
        return 4
                    
class MIMIC_CXR_EHR_RR_DN(Dataset):
    def __init__(self, args, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        
        self.CLASSES = CLASSES
        if 'radiology' in args.labels_set:
            self.CLASSES = R_CLASSES
        
        self.metadata_with_labels = metadata_with_labels
        self.discharge_notes_paired = self.metadata_with_labels['discharge_text'].values
        self.radiology_notes_paired = self.metadata_with_labels['radiology_text'].values
        if 'CXR' in args.modalities:
            self.time_diff = self.metadata_with_labels.time_diff
            self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        # self.lower = self.metadata_with_labels.lower
        # self.upper = self.metadata_with_labels.upper
        
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds
        self.args = args
        self.split = split
        self.data_ratio = self.args.data_ratio 
      
        self.paired_times= (self.metadata_with_labels['period_length'].values)
        self.ehr_paired_list = list(zip(self.ehr_files_paired, self.paired_times))

                
        if split=='test':
            self.data_ratio =  1.0
        elif split == 'val':
            self.data_ratio =  0.0
        

    def __getitem__(self, index):
        # lower = self.metadata_with_labels.iloc[index].lower
        # upper = self.metadata_with_labels.iloc[index].upper
        discharge_note = self.discharge_notes_paired[index]
        radiology_note = self.radiology_notes_paired[index] 
        age = age_to_bucket(self.metadata_with_labels.iloc[index]['age'])
        gender = GENDER[self.metadata_with_labels.iloc[index]['gender']]  # Numeric gender
        ethnicity = ETHNICITY[self.metadata_with_labels.iloc[index]['ethnicity']]
        hadm_id = self.metadata_with_labels.iloc[index]['hadm_id']
        if self.args.data_pairs == 'paired':
            # Determine the appropriate EHR DataFrame based on the task
            if self.args.task == 'decompensation' or self.args.task == 'length-of-stay':
                ehr_df = self.ehr_paired_list
            else:
                ehr_df = self.ehr_files_paired
        
            # Initialize labels and data
            ehr_data, labels_ehr = None, None
            cxr_data, labels_cxr = None, None
        
            # Handle EHR data loading
            if 'EHR' in self.args.modalities:
                ehr_data, labels_ehr = self.ehr_ds.__getitem__(ehr_df[index])#, lower, upper)
            else:
                ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
        
            # Handle CXR data loading
            if 'CXR' in self.args.modalities:
                cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                cxr_data, labels_cxr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
        
        if self.args.data_pairs == 'partial':
            ehr_file = self.metadata_with_labels['stay'].values[index]
            ehr_data, labels_ehr = self.ehr_ds.__getitem__(ehr_file)
            cxr_file = self.metadata_with_labels['dicom_id'].values[index]
            if pd.isna(cxr_file) or cxr_file == '' or not isinstance(cxr_file, str):
                cxr_data, labels_cxr = None, np.zeros(self.args.num_classes)
            else:
                cxr_data, labels_cxr = self.cxr_ds.__getitem__(cxr_file)
        
        return ehr_data, cxr_data, discharge_note, radiology_note, labels_ehr, labels_cxr, age, gender, ethnicity, hadm_id


    
    def __len__(self):
        if self.args.task == 'decompensation' or self.args.task == 'length-of-stay':
            ehr_df = self.ehr_paired_list
        else:
            ehr_df = self.ehr_files_paired
        if 'paired' in self.args.data_pairs:
            return len(self.ehr_files_paired)
        elif self.args.data_pairs == 'partial':
            # You need to decide what the dataset length should be in 'partial' mode.
            # For example, if you want to use the number of paired EHR files:
            return len(self.ehr_files_all)

        
def loadmetadata(args, discharge_notes, radiology_reports, cxr_reports):
    data_dir = args.cxr_data_dir
    cxr_metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_dir}/root/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime', 'hadm_id', 'age', 'ethnicity', 'gender']
    
    print(f"Initial CXR Metadata size: {cxr_metadata.shape}")
    print(f"Initial ICU Stay Metadata size: {icu_stay_metadata.shape}")

    cxr_merged_icustays = pd.DataFrame()
    print(args.task)
    if args.data_pairs == 'paired':
        if 'EHR' in args.modalities and 'CXR' in args.modalities:
            # Merge EHR and CXR data
            cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='inner', on='subject_id')
            cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
            cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)
            print(f"After merging CXR and ICU data (paired, EHR + CXR): {cxr_merged_icustays.shape}")
        elif 'EHR' in args.modalities:
            cxr_merged_icustays = icu_stay_metadata[columns]
            cxr_merged_icustays['StudyDateTime'] = None
            cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
            cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)
        elif 'CXR' in args.modalities:
            cxr_merged_icustays = cxr_metadata
            cxr_merged_icustays['intime'] = None
            cxr_merged_icustays['outtime'] = None
    
        if 'CXR' in args.modalities:
            cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
            cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str), format="%Y%m%d %H%M%S")
    
            if 'EHR' in args.modalities:
                cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
                cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)
        
                cxr_merged_icustays['time_diff'] = cxr_merged_icustays.StudyDateTime - cxr_merged_icustays.intime
                cxr_merged_icustays['time_diff'] = cxr_merged_icustays['time_diff'].apply(lambda x: np.round(x.total_seconds() / 60 / 60, 3))
        
                cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays.outtime - cxr_merged_icustays.intime
                cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays['full_stay_time'].apply(lambda x: np.round(x.total_seconds() / 60 / 60, 3))
    
        if 'RR' in args.modalities or 'DN' in args.modalities or 'CXRR' in args.modalities:
            dsrr_merge_columns = ['subject_id', 'hadm_id']
            dsrr_columns = ['subject_id', 'hadm_id','charttime', 'text']
    
        if 'DN' in args.modalities:
            cxr_merged_icustays = cxr_merged_icustays.merge(discharge_notes[dsrr_columns], how='left', on=dsrr_merge_columns)
            cxr_merged_icustays.rename(columns={'text': 'discharge_text'}, inplace=True)
            cxr_merged_icustays.rename(columns={'charttime': 'discharge_charttime'}, inplace=True)
            cxr_merged_icustays['discharge_charttime'] = pd.to_datetime(cxr_merged_icustays['discharge_charttime'])
        else:
            cxr_merged_icustays['discharge_text'] = None
    
        if 'RR' in args.modalities:
            cxr_merged_icustays = cxr_merged_icustays.merge(radiology_reports[dsrr_columns], how='left', on=dsrr_merge_columns)
            cxr_merged_icustays.rename(columns={'text': 'radiology_text'}, inplace=True)
            cxr_merged_icustays.rename(columns={'charttime': 'radiology_charttime'}, inplace=True)
            cxr_merged_icustays['radiology_charttime'] = pd.to_datetime(cxr_merged_icustays['radiology_charttime'])
            print(f"After merging with Radiology Reports: {cxr_merged_icustays.shape}")
        else:
            cxr_merged_icustays['radiology_text'] = None
        
        cxr_merged_icustays_during = cxr_merged_icustays
        if args.task == 'decompensation' or args.task == 'length-of-stay':
            train_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/train_listfile.csv')
            train_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            test_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/test_listfile.csv')
            test_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            val_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/val_listfile.csv')
            val_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            listfile = train_listfile.append(test_listfile)
            listfile = listfile.append(val_listfile)
            listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    
            columns2 = ['subject_id', 'endtime']
            listfile['subject_id'] = listfile['subject_id'].astype('int64')
            cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns2], how='inner', on='subject_id')
            cxr_merged_icustays.endtime = pd.to_datetime(cxr_merged_icustays.endtime)
            if 'CXR' in args.modalities:    
                cxr_merged_icustays_during = cxr_merged_icustays.loc[
                    ((cxr_merged_icustays.StudyDateTime >= cxr_merged_icustays.intime) & (cxr_merged_icustays.StudyDateTime <= cxr_merged_icustays.endtime))]
            if 'DN' in args.modalities:
                cxr_merged_icustays_during = cxr_merged_icustays_during.loc[
                    ((cxr_merged_icustays_during['discharge_charttime'] >= cxr_merged_icustays_during.intime)&(cxr_merged_icustays_during['discharge_charttime'] <= cxr_merged_icustays_during.endtime))]
            if 'RR' in args.modalities:
                    cxr_merged_icustays_during = cxr_merged_icustays_during.loc[
                    ((cxr_merged_icustays_during['radiology_charttime'] >= cxr_merged_icustays_during.intime)&(cxr_merged_icustays_during['radiology_charttime'] <= cxr_merged_icustays_during.endtime))]
                    
        if args.task == 'in-hospital-mortality':
            print("this is mortality")
            end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)
            #print("end_time:",end_time)
            if 'CXR' in args.modalities:
                cxr_merged_icustays_during = cxr_merged_icustays.loc[
                    ((cxr_merged_icustays.StudyDateTime >= cxr_merged_icustays.intime) & (cxr_merged_icustays.StudyDateTime <= end_time))]
            if 'RR' in args.modalities:
                cxr_merged_icustays_during = cxr_merged_icustays_during.loc[
                    ((cxr_merged_icustays_during['radiology_charttime'] >= cxr_merged_icustays_during.intime)&(cxr_merged_icustays_during['radiology_charttime'] <= cxr_merged_icustays_during.intime + pd.DateOffset(hours=48)))]
                
                group_columns = [col for col in cxr_merged_icustays_during.columns if col not in ['radiology_text', 'radiology_charttime']]
                print(group_columns)
                cxr_merged_icustays_during['radiology_text'] = cxr_merged_icustays_during['radiology_text'].fillna('').astype(str)
                for col in group_columns:
                    if cxr_merged_icustays_during[col].dtype == 'object':  # For categorical/text columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna('')
                    elif 'datetime' in str(cxr_merged_icustays_during[col].dtype):  # For datetime columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(pd.Timestamp('1900-01-01'))
                    else:  # For numeric columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(0)
                cxr_merged_icustays_during = cxr_merged_icustays_during.groupby(group_columns, as_index=False).agg({'radiology_text': lambda texts: ' '.join(texts),'radiology_charttime': 'max'})
            print("Columns in cxr_merged_icustays:", cxr_merged_icustays.columns)
            print("Columns in cxr_merged_icustays_during:", cxr_merged_icustays_during.columns)
            print(f"After task-specific filtering for In-Hospital Mortality: {cxr_merged_icustays_during.shape}")           
                
        if args.task == 'phenotyping' or args.task == 'readmission' or args.task == 'radiology':
            end_time = cxr_merged_icustays.outtime
            if 'RR' in args.modalities:
                # cxr_merged_icustays_during = cxr_merged_icustays_during.loc[
                #     ((cxr_merged_icustays_during['radiology_charttime'] >= cxr_merged_icustays_during.intime)&(cxr_merged_icustays_during['radiology_charttime'] <= cxr_merged_icustays_during.outtime))]
                group_columns = [col for col in cxr_merged_icustays_during.columns if col not in ['radiology_text', 'radiology_charttime']]
                print(group_columns)
                for col in group_columns:
                    if cxr_merged_icustays_during[col].dtype == 'object':  # For categorical/text columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna('')
                    elif 'datetime' in str(cxr_merged_icustays_during[col].dtype):  # For datetime columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(pd.Timestamp('1900-01-01'))
                    else:  # For numeric columns
                        cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(0)
                cxr_merged_icustays_during = cxr_merged_icustays_during.groupby(group_columns, as_index=False).agg({'radiology_text': lambda texts: ' '.join(texts),'radiology_charttime': 'max'})
    
        if 'CXR' in args.modalities:
            cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
            print(f"After filtering for AP-only CXR views: {cxr_merged_icustays_AP.shape}")
        else:
            cxr_merged_icustays_AP = cxr_merged_icustays_during

    elif args.data_pairs == 'partial':
        cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='right', on='subject_id')
        
        print("After merge with ICU stays:", cxr_merged_icustays.shape)
        print(cxr_merged_icustays[['StudyDate', 'StudyTime']].head(10))
        
        cxr_merged_icustays['StudyDate'] = cxr_merged_icustays['StudyDate'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) else None
        )
        
        
        cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' if pd.notna(x) else None)
        cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(
            cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str),
            format="%Y%m%d %H%M%S",
            errors='coerce'  # Invalid parsing will be set to NaT
        )
        
        print("After processing:", cxr_merged_icustays.shape)
        print(cxr_merged_icustays[['StudyDate', 'StudyTime']].head(10))
        
        # # Step 1: Clean StudyDate (remove decimals, convert to string)
        # cxr_merged_icustays['StudyDate'] = cxr_merged_icustays['StudyDate'].apply(
        #     lambda x: str(int(float(x))) if pd.notna(x) else None
        # )
        
        # # Step 2: Clean StudyTime (remove decimals, pad to 6 digits)
        # cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(
        #     lambda x: f"{int(float(x)):06}" if pd.notna(x) else None
        # )
        
        # # Step 3: Safely combine StudyDate and StudyTime
        # cxr_merged_icustays['CombinedDateTime'] = cxr_merged_icustays.apply(
        #     lambda row: f"{row['StudyDate']} {row['StudyTime']}" if pd.notna(row['StudyDate']) and pd.notna(row['StudyTime']) else None,
        #     axis=1
        # )
        
        # # Step 4: Convert to datetime
        # cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(
        #     cxr_merged_icustays['CombinedDateTime'],
        #     format="%Y%m%d %H%M%S",
        #     errors='coerce'
        # )
        
        
        
        print(cxr_merged_icustays['StudyDateTime'].head(10))
        #cxr_merged_icustays.loc[cxr_merged_icustays['StudyTime'].isna(), 'StudyDateTime'] = pd.NaT
        print("Count of NaT in StudyDateTime after formatting:", cxr_merged_icustays['StudyDateTime'].isna().sum())

        
        cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
        cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)

        cxr_merged_icustays['time_diff'] = cxr_merged_icustays.StudyDateTime - cxr_merged_icustays.intime
        cxr_merged_icustays['time_diff'] = cxr_merged_icustays['time_diff'].apply(lambda x: np.round(x.total_seconds() / 60 / 60, 3) if pd.notna(x) else x)

        cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays.outtime - cxr_merged_icustays.intime
        cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays['full_stay_time'].apply(lambda x: np.round(x.total_seconds() / 60 / 60, 3) if pd.notna(x) else x)
        dsrr_merge_columns = ['subject_id', 'hadm_id']
        dsrr_columns = ['subject_id', 'hadm_id','charttime', 'text']
        cxr_merged_icustays = cxr_merged_icustays.merge(discharge_notes[dsrr_columns], how='left', on=dsrr_merge_columns)
        cxr_merged_icustays.rename(columns={'text': 'discharge_text'}, inplace=True)
        cxr_merged_icustays.rename(columns={'charttime': 'discharge_charttime'}, inplace=True)
        cxr_merged_icustays['discharge_charttime'] = pd.to_datetime(cxr_merged_icustays['discharge_charttime'])
        cxr_merged_icustays = cxr_merged_icustays.merge(radiology_reports[dsrr_columns], how='left', on=dsrr_merge_columns)
        cxr_merged_icustays.rename(columns={'text': 'radiology_text'}, inplace=True)
        cxr_merged_icustays.rename(columns={'charttime': 'radiology_charttime'}, inplace=True)
        cxr_merged_icustays['radiology_charttime'] = pd.to_datetime(cxr_merged_icustays['radiology_charttime'])
        cxr_merged_icustays_during = cxr_merged_icustays
    
        if args.task == 'decompensation' or args.task == 'length-of-stay':
            train_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/train_listfile.csv')
            train_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            test_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/test_listfile.csv')
            test_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            val_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/val_listfile.csv')
            val_listfile.columns = ['stay', 'period_length', 'stay_id', 'y_true', 'intime', 'endtime']
            listfile = train_listfile.append(test_listfile)
            listfile = listfile.append(val_listfile)
            listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    
            columns2 = ['subject_id', 'endtime']
            listfile['subject_id'] = listfile['subject_id'].astype('int64')
            cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns2], how='inner', on='subject_id')
            cxr_merged_icustays.endtime = pd.to_datetime(cxr_merged_icustays.endtime)
           
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime,  'dicom_id'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime,  'radiology_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime, 'StudyDateTime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime, 'radiology_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, ['StudyDateTime', 'dicom_id']] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime < cxr_merged_icustays.intime, 'discharge_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime,'radiology_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, 'StudyDateTime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime < cxr_merged_icustays.intime, 'discharge_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime, 'radiology_charttime'] = pd.NaT
            
        if args.task == 'in-hospital-mortality':
            cxr_merged_icustays.endtime = cxr_merged_icustays.intime + pd.DateOffset(hours=48)
            
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime, 'dicom_id'] = None
            #cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime, 'radiology_text'] = None
            print("Count NaNs in StudyDateTime before setting:", cxr_merged_icustays['StudyDateTime'].isna().sum())
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime, 'StudyDateTime'] = None
            print("Count NaNs in StudyDateTime after setting post-endtime to NaT:", cxr_merged_icustays['StudyDateTime'].isna().sum())
            #cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime, 'radiology_charttime'] = None
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, 'dicom_id'] = None
            cxr_merged_icustays['discharge_text'] = None
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime, 'radiology_text'] = None
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, 'StudyDateTime'] = None
            print("final happening",cxr_merged_icustays['StudyDateTime'].isna().sum())
            cxr_merged_icustays['discharge_charttime'] = None
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime, 'radiology_charttime'] = None
            
            
            cxr_merged_icustays.loc[cxr_merged_icustays['StudyDateTime'].isna(), 'dicom_id'] = None
            cxr_merged_icustays.loc[cxr_merged_icustays['radiology_charttime'].isna(), 'radiology_text'] = None
            
            
            print("this is mortality")
            
            
            # Replace NaN in specific columns with arbitrary values
            cxr_merged_icustays['radiology_text'] = cxr_merged_icustays['radiology_text'].fillna('')
            cxr_merged_icustays['radiology_charttime'] = cxr_merged_icustays['radiology_charttime'].fillna(pd.Timestamp('1900-01-01'))
            
            cxr_merged_icustays_during = cxr_merged_icustays

            group_columns = [col for col in cxr_merged_icustays_during.columns if col not in ['radiology_text', 'radiology_charttime']]
            print(group_columns)
            for col in group_columns:
                if cxr_merged_icustays_during[col].dtype == 'object':  # For categorical/text columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna('')
                elif 'datetime' in str(cxr_merged_icustays_during[col].dtype):  # For datetime columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(pd.Timestamp('1900-01-01'))
                else:  # For numeric columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(0)
            cxr_merged_icustays_during = cxr_merged_icustays_during.groupby(group_columns, as_index=False).agg({'radiology_text': lambda texts: ' '.join(texts),'radiology_charttime': 'max'})
            print("Columns in cxr_merged_icustays:", cxr_merged_icustays.columns)
            print("Columns in cxr_merged_icustays_during:", cxr_merged_icustays_during.columns)
            print(f"After task-specific filtering for In-Hospital Mortality: {cxr_merged_icustays_during.shape}")
            
    
        if args.task == 'phenotyping' or args.task == 'readmission':
            cxr_merged_icustays.endtime = cxr_merged_icustays.outtime
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime, 'dicom_id'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime, 'radiology_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime > cxr_merged_icustays.endtime, 'StudyDateTime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime > cxr_merged_icustays.endtime, 'discharge_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime > cxr_merged_icustays.endtime, 'radiology_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, 'dicom_id'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime < cxr_merged_icustays.intime, 'discharge_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime, 'radiology_text'] = np.nan
            cxr_merged_icustays.loc[cxr_merged_icustays.StudyDateTime < cxr_merged_icustays.intime, 'StudyDateTime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.discharge_charttime < cxr_merged_icustays.intime, 'discharge_charttime'] = pd.NaT
            cxr_merged_icustays.loc[cxr_merged_icustays.radiology_charttime < cxr_merged_icustays.intime, 'radiology_charttime'] = pd.NaT
            
            # Replace NaN in specific columns with arbitrary values
            cxr_merged_icustays['radiology_text'] = cxr_merged_icustays['radiology_text'].fillna('')
            cxr_merged_icustays['radiology_charttime'] = cxr_merged_icustays['radiology_charttime'].fillna(pd.Timestamp('1900-01-01'))
            
            cxr_merged_icustays_during = cxr_merged_icustays
            
            group_columns = [col for col in cxr_merged_icustays_during.columns if col not in ['radiology_text', 'radiology_charttime']]
            print(group_columns)
            for col in group_columns:
                if cxr_merged_icustays_during[col].dtype == 'object':  # For categorical/text columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna('')
                elif 'datetime' in str(cxr_merged_icustays_during[col].dtype):  # For datetime columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(pd.Timestamp('1900-01-01'))
                else:  # For numeric columns
                    cxr_merged_icustays_during[col] = cxr_merged_icustays_during[col].fillna(0)
            cxr_merged_icustays_during = cxr_merged_icustays_during.groupby(group_columns, as_index=False).agg({'radiology_text': lambda texts: ' '.join(texts),'radiology_charttime': 'max'})
            print("Columns in cxr_merged_icustays:", cxr_merged_icustays.columns)
            print("Columns in cxr_merged_icustays_during:", cxr_merged_icustays_during.columns)
            print(f"After task-specific filtering for In-Hospital Mortality: {cxr_merged_icustays_during.shape}")
        
        cxr_merged_icustays_during.loc[cxr_merged_icustays_during['ViewPosition'] != 'AP', 'dicom_id'] = None
        cxr_merged_icustays_AP = cxr_merged_icustays_during
        
    groups = cxr_merged_icustays_AP.groupby('stay_id')

    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    print(f"After selecting the latest CXR per stay: {groups.shape}")

    return groups


def load_cxr_ehr_rr_dn(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds):
    discharge_notes=pd.read_csv('/scratch/baj321/MIMIC-Note/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv')
    radiology_reports=pd.read_csv('/scratch/baj321/MIMIC-Note/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv')
    #cxr_reports = pd.read_csv("/scratch/baj321/cxr_reports.csv")
    cxr_reports= None

    cxr_merged_icustays = loadmetadata(args, discharge_notes, radiology_reports, cxr_reports) 

    splits_labels_train = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/test_listfile.csv')
    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')
    
    train_ds = MIMIC_CXR_EHR_RR_DN(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    val_ds = MIMIC_CXR_EHR_RR_DN(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    test_ds = MIMIC_CXR_EHR_RR_DN(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')
    
    
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=lambda batch: my_collate(batch, args), pin_memory=True, num_workers=16, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=lambda batch: my_collate(batch, args), pin_memory=True, num_workers=16, drop_last=False)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=lambda batch: my_collate(batch, args), pin_memory=True, num_workers=16, drop_last=False)
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    return train_dl, val_dl, test_dl


def printPrevalence(merged_file, args):
    if args.labels_set == 'pheno':
        total_rows = len(merged_file)
        print(merged_file[CLASSES].sum()/total_rows)
    else:
        total_rows = len(merged_file)
        print(merged_file['y_true'].value_counts())
    # import pdb; pdb.set_trace()

    
# def my_collate(batch, args):
#     x = [item[0] for item in batch]
#     pairs = [False if item[1] is None else True for item in batch]
    
#     # Change: Ensure that all elements passed to torch.stack are Tensors
#     img = torch.stack([torch.zeros(3, 384, 384) if item[1] is None else (item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])) for item in batch])
    
#     x, seq_length = pad_zeros(x, args = args)
#     discharge_note = [
#         '' if item[2] is None else str(item[2])
#         for item in batch
#     ]
#     radiology_note = [
#         '' if item[3] is None else str(item[3])
#         for item in batch
#     ]
#     targets_ehr = np.array([item[4] for item in batch])
#     if args.data_pairs == "paired":
#         targets_cxr = torch.stack([torch.zeros(14) if item[5] is None else (item[5] if isinstance(item[5], torch.Tensor) else torch.tensor(item[5])) for item in batch])
#     else:
#         targets_cxr = None
#     # if len(batch[0]) > 6:  # Adjust this condition based on what constitutes an appropriate batch size
#     #     age = torch.Tensor(np.array([item[6] for item in batch])).unsqueeze(1)
#     #     gender = torch.Tensor(np.array([item[7] for item in batch])).unsqueeze(1)
#     #     ethnicity = torch.Tensor(np.array([item[8] for item in batch]))
#     #     return [x, img, discharge_note, radiology_note, targets_ehr, targets_cxr, seq_length, pairs, age, gender, ethnicity]
#     # else:
#     return [x, img, discharge_note, radiology_note, targets_ehr, targets_cxr, seq_length, pairs]

def my_collate(batch, args):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    
    # Change: Ensure that all elements passed to torch.stack are Tensors
    img = torch.stack([torch.zeros(3, 384, 384) if item[1] is None else (item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])) for item in batch])
    
    x, seq_length = pad_zeros(x, args = args)
    
    discharge_note = [
        "" if pd.isna(item[2]) else str(item[2])
        for item in batch
    ]
    radiology_note = [
        "" if pd.isna(item[3]) else str(item[3])
        for item in batch
    ]
    targets_ehr = np.array([item[4] for item in batch])
    if args.data_pairs == "paired":
        targets_cxr = torch.stack([torch.zeros(14) if item[5] is None else (item[5] if isinstance(item[5], torch.Tensor) else torch.tensor(item[5])) for item in batch])
    else:
        targets_cxr = None
    
    age = torch.Tensor(np.array([item[6] for item in batch])).unsqueeze(1)
    gender = torch.Tensor(np.array([item[7] for item in batch])).unsqueeze(1)
    ethnicity = torch.Tensor(np.array([item[8] for item in batch]))
    hadm_id = torch.Tensor(np.array([item[9] if item[9] is not None else 0 for item in batch]))
    
    return [x, img, discharge_note, radiology_note, targets_ehr, targets_cxr, seq_length, pairs, age, gender, ethnicity, hadm_id]

def pad_zeros(arr, args, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    if args.task == "in-hospital-mortality":
        min_length=None
    else:
        min_length=2442
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length