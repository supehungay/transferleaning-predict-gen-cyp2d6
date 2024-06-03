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
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

import os
import sys
import joblib
import glob
sys.path.append('../src/')
import vcf2onehot


class TransferLearningModel(vcf2onehot.VCF2Onehot):
    def __init__(self, train_path, test_path, path_model_pretrain, num_model = 7):
        self.train_path = train_path
        self.test_path = test_path
        self.path_model = path_model_pretrain
        self.now = datetime.now().strftime('%d-%m-%Y_%H-%M')
        self.num_model = num_model

        self.dataset_trainning = self.load_data(path_data=self.train_path)
        self.dataset_evalution = self.load_data(path_data=self.test_path)
        
        
    def load_data(self, path_data: str) -> tuple:
        X = None
        y = None
        data_loaded = joblib.load(path_data)
        X = data_loaded['X']
        y = data_loaded['y']
        
        all_stars = np.array([s.split('_')[1] for s in data_loaded['sample_names']]) # lấy ra star alen: 10, 1, 2, ...
        stars, idx = np.unique(all_stars, return_index=True)
        sample_mask = np.isin(all_stars, all_stars[idx]) # Đánh dấu những star allele của idx trong all_stars
        stars_001 = np.array([s for s in data_loaded['sample_names'][sample_mask] if s.split('_')[-1] == '001']) 
        mark_001 = np.isin(data_loaded['sample_names'], stars_001)
        X_001, y_001 = X[mark_001], y[mark_001]
        
        dataset = [(X, y), (X_001, y_001)]
        
        return dataset

    def get_model_pretrained(self):
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
        
        return model
    
    def train_model(self):
        _train_ds = tf.data.Dataset.from_tensor_slices(self.dataset_trainning[0])
        train_ds = _train_ds.repeat().shuffle(self.dataset_trainning[0][1].shape[0], reshuffle_each_iteration=True).batch(32).prefetch(buffer_size=10)

        for i in range(self.num_model):
            print(f"tfl_model_{i}.model.h5")
            model = self.build_and_fit(train_ds)
            model.save(f'../model/FINAL_MODEL/{self.now}/tfl_model_{i}.model.h5')
    
    def evaluate_model(self):
        models = glob.glob(f'../model/FINAL_MODEL/{self.now}/' + "*.model.h5")
        evaluates = [] 
        X_test, y_test = self.dataset_evalution[0]
        for m in models: 
            model = load_model(m)       
            evaluate = model.evaluate(X_test, y_test)
            evaluates.append(evaluate)
            
        print(f'Loss_test and Accuracy_test: {np.array(evaluates).mean(axis=0)}; evaluate on dataset_evalution')
        
    
def main():
    TRAIN_PATH = '../data/final_model/train/train_data.joblib'
    TEST_PATH = '../data/final_model/test/test_data.joblib'
    PATH_MODEL = '../save_model/FinalModel/final_25-03-2024_02-32/model.h5'
    
    num_model = 7
    
    transfer_model = TransferLearningModel(train_path=TRAIN_PATH, test_path=TEST_PATH, path_model_pretrain=PATH_MODEL, num_model = num_model)
    transfer_model.train_model()
    transfer_model.evaluate_model()
    
if __name__ == '__main__':
    main()