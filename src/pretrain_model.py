import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import joblib
import glob
from datetime import datetime
import os

class PretrainModel:
    def __init__(self, train_path, test_path):
        # self.X = None
        # self.y = None
        self.train_path = train_path
        self.test_path = test_path
        # self.input_shape = None
        self.model = None
        

    def label2onehot(self, label: np.array) -> np.array:
        thresholds = [0.25, 0.75, 1.25, 1.75]
    
        categorical_labels = np.digitize(label, thresholds)
    
        one_hot_encoder = to_categorical(categorical_labels)
        
        return one_hot_encoder
    
    def load_data(self, path_data: str) -> tuple:
        X = None
        y = None
        all_path = sorted(glob.glob(f'{path_data}/*'))
        for count, path in enumerate(all_path):
            # if count == 10:
            #     break
            print(f"{count}. {path.split('/')[-1].split('.')[0]}")
            data_loaded = joblib.load(path)
            if X is None:
                X = data_loaded['X']
                y = data_loaded['activate_score']
            else:
                X = np.concatenate((X, data_loaded['X']), axis=0)
                y = np.concatenate((y, data_loaded['activate_score']), axis=0)
    
        y = y.reshape(-1, 1)
        y = self.label2onehot(y)
    
        return shuffle(X, y, random_state=10)
    
    
    def build_model(self, input_shape: tuple):
        inputs = Input(shape=input_shape, name="data")
        
        layer = Conv1D(70, kernel_size=19, strides=5, input_shape = (14868, 13), activation='linear', name = "conv1d_1")(inputs)
        layer = BatchNormalization(name="batch_1")(layer)
        # layer = ReLU(name="relu_1")(layer)
        layer = Activation(activation='relu', name=f'activation_{1}')(layer)
        
        layer = MaxPooling1D(pool_size=3, strides=3, name="maxpooling_1")(layer)
        layer = Conv1D(46, kernel_size=11, strides=5, activation='linear', name = "conv1d_2")(layer)
        layer = BatchNormalization(name="batch_2")(layer)
        # layer = ReLU(name="relu_2")(layer)
        layer = Activation(activation='relu', name=f'activation_{2}')(layer)
        
        layer = MaxPooling1D(pool_size=4, strides=4, name="maxpooling_2")(layer)
        layer = Conv1D(46, kernel_size=7, strides=5, activation='linear', name = "conv1d_3")(layer)
        layer = BatchNormalization(name="batch_3")(layer)
        # layer = ReLU(name="relu_3")(layer)
        layer = Activation(activation='relu', name=f'activation_{3}')(layer)
        
        layer = MaxPooling1D(pool_size=4, strides=4, name="maxpooling_3")(layer)
        layer = Flatten(name="flatten_3")(layer)
        layer = Dense(32, activation='relu', name="dense_4")(layer)
        layer = Dropout(rate=0.03, name="dropout_4")(layer)
        
        outputs = tf.keras.layers.Dense(5, activation='softmax', name="dense_5")(layer)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN_model')
        return model

    def train_model(self):
        X_train, y_train = self.load_data(path_data=self.train_path)
        input_shape = X_train.shape[1:]
        self.model = self.build_model(input_shape)
        # self.input_shape = input_shape
        print(f'X_train: {X_train.shape}; y_train: {y_train.shape}; input_shape: {input_shape}')
    
        # model = build_model()
        self.model.summary()
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        now = datetime.now().strftime('%d-%m-%Y_%H-%M')
        
        model_name = os.path.basename(__file__).split('.')[0]
        
        checkpoint_callback = ModelCheckpoint(filepath=f"../model/PRETRAIN_MODEL/ModelCheckPoint/{model_name}_{now}/" + "model.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5",
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=False,
                                            verbose=1)
        tensorboard_callback = TensorBoard(log_dir=f"../model/PRETRAIN_MODEL/TensorBoard/{model_name}_{now}/logs")
    
        folder_logger_path = f"../model/PRETRAIN_MODEL/CSVLogger/{model_name}_{now}"
    
        if os.path.exists(folder_logger_path) and os.path.isdir(folder_logger_path):
            os.rmdir(folder_logger_path)
        os. makedirs(folder_logger_path)
    
        csv_logger_callback = CSVLogger(f"{folder_logger_path}/training.log")
        
        self.model.fit(X_train, y_train, 
                epochs=10,
                validation_split=.15,  
                batch_size=64,
                callbacks=[checkpoint_callback,
                            tensorboard_callback,
                            csv_logger_callback]
                )
        
        self.model.save(f"../model/PRETRAIN_MODEL/FinalModel/{model_name}_{now}/model.h5")


    def evalute_model(self):
        X_test, y_test = self.load_data(self.test_path)
        evaluate_model = self.model.evaluate(X_test, y_test)
        print(f'Loss_test and Accuracy_test: {evaluate_model}; evaluate on 10k dataset')
        

def main():
    TRAIN_PATH = '../data/pretrained_model/train'
    TEST_PATH = '../data/pretrained_model/test'  
    model = PretrainModel(TRAIN_PATH, TEST_PATH)
    # model.load_data()
    # model.build_model()
    model.train_model()
    model.evalute_model()

if __name__ == '__main__':
    main()
    
