import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, Dropout
from keras.models import Model
import tensorflow as tf

from keras.models import load_model
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime
import os

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

import sys
import glob
import joblib
sys.path.append('../src/')
# import vcf2onehot
from vcf2onehot import VCF2Onehot


class PrepareData:
    def __init__(self, pre_data_path, pre_label_path, tranf_data_path, tranf_label_path, cerated_func) -> None:
        self.pre_data_path = pre_data_path
        self.pre_label_path = pre_label_path
        self.tranf_data_path = tranf_data_path
        self.tranf_label_path = tranf_label_path
        self.cerated_func = cerated_func
        
        self.prepare_pretrain_model(data_path=self.pre_data_path, label_path=self.pre_label_path)
        self.process_label_data_transfer(star_sample=self.tranf_data_path, curated_func=self.cerated_func, save_label=tranf_label_path)
        self.prepare_transfer_model(data_path=self.tranf_data_path, label_path=self.tranf_label_path)
    
    def prepare_pretrain_model(self, data_path, label_path):
        print('preparing data for pretrain model...')
        
        batch_data = sorted(glob.glob(data_path))
        batch_label = sorted(glob.glob(label_path))
        
        process_data = VCF2Onehot(data_path=batch_data, label_path=batch_label)
        
        n_batch = 12
        n_batch_train = 10
        
        print(f'create {n_batch} batch: {n_batch_train} batch train and {n_batch - n_batch_train} batch test')
        
        for i in range(n_batch):
            activate_score = []
            # Generate the seq data
            vcf = batch_data[i]
            seqs = process_data.build_seqs(vcf=vcf)
            # Create the data object
            data = process_data.format_seqs(seqs)

            with open(f'{batch_label[i]}') as f:
                for line in f:
                    activate_score.append(line.split(',')[-2])
            
            data["activate_score"] = np.array(activate_score, dtype=np.float64)
            
            file_name = vcf.split('\\')[-1].split('.')[0]
            
            if i < n_batch_train:
                joblib.dump(data, f'../data/pretrained_model/train/{file_name}.joblib')
                print(f"save train batch {i + 1}...")
            else:
                joblib.dump(data, f'../data/pretrained_model/test/{file_name}.joblib')
                print(f'save test batch {(i - n_batch_train) + 1}...')
            
        print("Done!\n")
        
    def get_sample_names(self, star_sample) -> list:
        samples = []
        with open(star_sample) as f:
            for line in f:
                if line.startswith("#CHROM"):
                    samples = line.strip().split()[9:]
                    break
        return samples
    
    
    def process_label_data_transfer(self, star_sample, curated_func, save_label):
        print('processing label for transfer learning model...')
        
        alleles_function = pd.read_excel(curated_func, usecols=[0, 1])
        
        samples = self.get_sample_names(star_sample)
            
        labels = []
        for sample in samples:
            star = "*" + str(sample.split("_")[1])
            label = alleles_function[alleles_function["CYP2D6 Star Allele"] == star]["Curated Function"].values[0]

            if label == "Uncurated":
                no_function = None
                normal_function = None
            else:
                no_function = 0 if label == "No function" else 1
                normal_function = 1 if label == "Normal" else 0

            labels.append([sample, no_function, normal_function])
            
        label_df = pd.DataFrame(labels)
        label_df.to_csv(save_label, index=False, header=None)
        print("Saved labels to", save_label)
        print('Done!\n')
        
        
    def get_data(self, data_path, label_path):
        samples = self.get_sample_names(data_path)
        process_data = VCF2Onehot(data_path=data_path, label_path=label_path)
        
        seqs = process_data.build_seqs(data_path)
        data = process_data.format_seqs(seqs)
        data['y'] = pd.read_csv(label_path, header=None, index_col=0).loc[samples].values
        
        return data
    
    def split_data(self, data: dict) -> tuple:
        determined_samples = {} # lưu những mẫu đã xác định chức năng
        uncurated_samples ={} # lưu những mẫu chức năng chưa xác định chức năng
        
        mask = np.all(np.isnan(data['y']) == False, axis=1)
        
        determined_samples['X'] = data['X'][mask]
        determined_samples['y'] = data['y'][mask]
        determined_samples['sample_names'] = data['sample_names'][mask]

        uncurated_samples['X'] = data['X'][~mask]
        uncurated_samples['sample_names'] = data['sample_names'][~mask]

        uncurated_stars = np.array([s for s in uncurated_samples['sample_names'] if s.split('_')[-1] == '001']) # chỉ lấy những mẫu cs subalen = 001 để dự đoán
        uncurated_star_mask = np.isin(uncurated_samples['sample_names'], uncurated_stars) 
        
        uncurated_samples['sample_names'] = uncurated_samples['sample_names'][uncurated_star_mask]
        uncurated_samples['X'] = uncurated_samples['X'][uncurated_star_mask]
        
        return determined_samples, uncurated_samples
    
    def prepare_transfer_model(self, data_path, label_path):
        print('preparing data for transfer model...')
        
        data = self.get_data(data_path, label_path)
        
        determined_samples, uncurated_samples = self.split_data(data)
        joblib.dump(uncurated_samples, '../data/final_model/uncerated/uncerated.joblib')
        print(f'saved uncerated samples to ../data/final_model/uncerated/uncerated.joblib')
        
        
        all_stars = np.array([s.split('_')[1] for s in determined_samples['sample_names']]) # lấy ra star alen: 10, 1, 2, ...
        stars, idx = np.unique(all_stars, return_index=True)
        train_idx, test_idx = train_test_split(idx, stratify=determined_samples['y'][idx], test_size=24, random_state=10)
        
        sample_mask = np.isin(all_stars, all_stars[train_idx]) # Đánh dấu những star allele của train_idx trong all_stars

        # gồm tất cả các allele + suballele sử dụng trong quá trình trainning
        train_data = {}
        test_data = {}
        
        train_data['X'], train_data['y'], train_data['sample_names']= determined_samples['X'][sample_mask], determined_samples['y'][sample_mask], determined_samples['sample_names'][sample_mask]
        test_data['X'], test_data['y'], test_data['sample_names'] = determined_samples['X'][~sample_mask], determined_samples['y'][~sample_mask], determined_samples['sample_names'][~sample_mask]

        joblib.dump(train_data, f'../data/final_model/train/train_data.joblib')
        print(f"saved test data to ../data/final_model/train/train_data.joblib")
        
        joblib.dump(test_data, f'../data/final_model/test/test_data.joblib')
        print(f"saved test data to ../data/final_model/test/test_data.joblib")
        print('Done!\n')
        
def main():
    DATA_PATH1 = '../data/simulated_cyp2d6_diplotypes/*.vcf'
    LABEL_PATH1 = '../data/simulated_cyp2d6_diplotypes/*.csv'
    DATA_PATH2 = '../data/final_model/star_samples.vcf'
    LABEL_PATH2 = '../data/final_model/labels_alleles.csv'
    CERATED_FUNC = '../data/final_model/pcbi.1008399.s003.xlsx'
    
    prepare_data = PrepareData(pre_data_path=DATA_PATH1, pre_label_path=LABEL_PATH1, tranf_data_path=DATA_PATH2, tranf_label_path=LABEL_PATH2, cerated_func=CERATED_FUNC)
    
    # prepare_pretrain_model(data_path=data_path1, label_path=label_path1)

    # process_label_data_transfer(star_sample=DATA_PATH2, curated_func=CURATED_FUNC, save_label=LABEL_PATH2)
    
    # prepare_transfer_model(data_path=DATA_PATH2, label_path=LABEL_PATH2)
    
    
if __name__=="__main__":
    main()