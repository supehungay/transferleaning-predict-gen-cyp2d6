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
sys.path.append('../src/')
import vcf2onehot


class TransferLearningModel(vcf2onehot.VCF2Onehot):
    def __init__(self, star_sample, curated_function, path_label, path_model_pretrain):
        self.star_sample = star_sample
        self.curated_function = curated_function
        self.path_label = path_label
        self.path_model = path_model_pretrain
        self.now = datetime.now().strftime('%d-%m-%Y_%H-%M')
        
        self.data = self.get_data()
        self.determined_samples, self.uncurated_samples = self.split_data(self.data)
        self.dataset_trainning, self.dataset_evalution = self.get_dataset_train_eval(self.determined_samples)
        self.train_ds, self.test_ds = self.cvtoDataset(self.dataset_trainning)
        
        
    
    def get_sample_names(self) -> list:
        samples = []
        with open(self.star_sample) as f:
            for line in f:
                if line.startswith("#CHROM"):
                    samples = line.strip().split()[9:]
                    break
        return samples
    
    def get_data(self) -> dict:
        samples = self.get_sample_names()
        seqs = self.build_seqs(self.star_sample)
        data = self.format_seqs(seqs)
        data['y'] = pd.read_csv(self.path_label, header=None, index_col=0).loc[samples].values
        
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
                    

    def get_dataset_train_eval(self, determined_samples: dict) -> tuple:
        all_stars = np.array([s.split('_')[1] for s in determined_samples['sample_names']]) # lấy ra star alen
        stars, idx = np.unique(all_stars, return_index=True)
        train_idx, test_idx = train_test_split(idx, stratify=determined_samples['y'][idx], test_size=24, random_state=10)
        
        sample_mask = np.isin(all_stars, all_stars[train_idx])
        
        train_stars = np.array([s for s in determined_samples['sample_names'][sample_mask] if s.split('_')[-1] == '001'])
        train_mask = np.isin(determined_samples['sample_names'], train_stars)
            
        test_stars = np.array([s for s in determined_samples['sample_names'][~sample_mask] if s.split('_')[-1] == '001'])
        test_mask = np.isin(determined_samples['sample_names'], test_stars)

        # gồm tất cả các allele + suballele sử dụng trong quá trình trainning
        train_X_all_allele, test_X_all_allele = determined_samples['X'][sample_mask], determined_samples['X'][~sample_mask]
        train_y_all_allele, test_y_all_allele = determined_samples['y'][sample_mask], determined_samples['y'][~sample_mask]
        dataset_trainning = [(train_X_all_allele, train_y_all_allele), (test_X_all_allele, test_y_all_allele)]

        # chỉ bao gồm các allele 001 dùng để đánh giá
        train_X_star, test_X_star = determined_samples['X'][train_mask], determined_samples['X'][test_mask]
        train_y_star, test_y_star = determined_samples['y'][train_mask], determined_samples['y'][test_mask]
        dataset_evalution = [(train_X_star, train_y_star), (test_X_star, test_y_star)]
        
        return dataset_trainning, dataset_evalution
        
    
    def cvtoDataset(self, dataset_trainning):
        _train_ds = tf.data.Dataset.from_tensor_slices(dataset_trainning[0])
        train_ds = _train_ds.repeat().shuffle(dataset_trainning[0][1].shape[0], reshuffle_each_iteration=True).batch(32).prefetch(buffer_size=10)

        _test_ds = tf.data.Dataset.from_tensors(dataset_trainning[1])
        test_ds = _test_ds.prefetch(buffer_size=10)
        
        return train_ds, test_ds
    
    
    def get_model_pretrained(self):
        # PATH_MODEL = '../save_model/final_25-03-2024_02-32/model.h5'
        model = load_model(self.path_model)
        return model
    
    def create_transfer_model(self):
        model = self.get_model_pretrained()
        model.pop()
        model.pop()
        model.pop()
        model.trainable = False

        layer = Dense(units=32, activation='relu', name='dense5')(model.output)
        layer = Dropout(0.3, name='dropout_4')(layer)
        layer = Dense(units=1, activation='linear', name='dense6')(layer)
        outputs = Dense(units=2, activation='sigmoid')(layer)

        transfer_model = Model(inputs=model.input, outputs=outputs, name='transfer_model')
        
        return transfer_model
    
    def build_and_fit(self, dataset):

        model = self.create_transfer_model()
        model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.AUC()]
                )
        
        model.fit(dataset, 
                steps_per_epoch=128, 
                epochs=20, 
                verbose=True)
        
        # model.save(f"../model/FINAL_MODEL/{self.now}/model.h5")
        
        return model
    
    def train_model(self):
        num_model = 7
        for i in range(num_model):
            print(f"tfl_model_{i}.model.h5")
            model = self.build_and_fit(self.train_ds)
            model.save(f'../model/FINAL_MODEL/{self.now}/tfl_model_{i}.model.h5')
            
def main():
    STAR_SAMPLE = '../data/final_model/star_samples.vcf'
    CURATED_FUNCTION = '../data/final_model/pcbi.1008399.s003.xlsx'
    PATH_LABEL_SAVE = '../data/final_model/labels_alleles.csv'
    PRETRAIN_MODEL = '../save_model/final_25-03-2024_02-32/model.h5'
    
    transfer_model = TransferLearningModel(star_sample=STAR_SAMPLE, curated_function=CURATED_FUNCTION, path_label=PATH_LABEL_SAVE, path_model_pretrain=PRETRAIN_MODEL)
    transfer_model.train_model()
    
if __name__ == '__main__':
    main()