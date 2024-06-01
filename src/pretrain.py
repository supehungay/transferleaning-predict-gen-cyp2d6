import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
import joblib
import glob

from datetime import datetime
import os
from keras.models import load_model, model_from_json

def label2onehot(label: np.array) -> np.array:
    thresholds = [0.25, 0.75, 1.25, 1.75]

    categorical_labels = np.digitize(label, thresholds)

    one_hot_encoder = to_categorical(categorical_labels)
    
    return one_hot_encoder

def load_data() -> tuple:
    X = None
    y = None
    all_path = sorted(glob.glob('../data/input_data/*'))

    for count, path in enumerate(all_path):
        print(f"{count}. {path.split('/')[-1].split('.')[0]}")
        data_loaded = joblib.load(path)
        if X is None:
            X = data_loaded['X']
            y = data_loaded['activate_score']
        else:
            X = np.concatenate((X, data_loaded['X']), axis=0)
            y = np.concatenate((y, data_loaded['activate_score']), axis=0)
    
        if count == 50:
            break
        
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, random_state=10)        
    
    # y_train = y_train.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)
    
    y = y.reshape(-1, 1)
    y = label2onehot(y)
    
    return X, y

def call_trained_model():
    m = '../model/hubble2d6_0'

    json_file = open('%s.json' % m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    json_file.close()

    # Load the weights to the model
    model.load_weights("%s.model.h5" % m)
    
    return model


def create_transfer_model(num_classes=5):
    model = call_trained_model()
    
    for layer in model.layers:
        layer.trainable = False

    last_conv_layer = model.layers[-6].output

    layer = Dense(units=32, activation='linear', kernel_initializer='VarianceScaling', bias_initializer='Zeros', name='dense1')(last_conv_layer)
    layer = Activation(activation='relu', name=f'activation_{4}')(layer)
    layer = Dropout(0.3, name='dropout_4')(layer)

    layer = Dense(units=64, activation='relu', kernel_initializer='VarianceScaling', bias_initializer='Zeros', name='dense2')(layer)

    outputs = Dense(units=5, activation='softmax', kernel_initializer='VarianceScaling', bias_initializer='Zeros', name='final_denes')(layer)

    transfer_model = Model(inputs=model.input, outputs=outputs, name='cnn_model')
    
    return transfer_model


    


def main():
    X_train, X_test, y_train, y_test = load_data()
    input_shape = X_train.shape[1:]
    print(f'X_train: {X_train.shape}; X_test: {X_test.shape}; y_train: {y_train.shape}; y_test: {y_test.shape}, input_shape{input_shape}')

    model = model = create_transfer_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    now = datetime.now().strftime('%d-%m-%Y_%H-%M')
    checkpoint_callback = ModelCheckpoint(filepath=f"../model/ModelCheckPoint/{now}/" + "model.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5",
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        verbose=1)
    tensorboard_callback = TensorBoard(log_dir=f"../model/TensorBoard/{now}/logs")
    
    folder_logger_path = f"../model/CSVLogger/{now}"

    if os.path.exists(folder_logger_path) and os.path.isdir(folder_logger_path):
        os.rmdir(folder_logger_path)
    os. makedirs(folder_logger_path)

    csv_logger_callback = CSVLogger(f"{folder_logger_path}/training.log")
    
    model.fit(X_train, y_train, 
            epochs=10,
            validation_split=.15,  
            batch_size=64,
            callbacks=[checkpoint_callback,
                        tensorboard_callback,
                        csv_logger_callback]
            )
    
    model.save(f"../model/FinalModel/{now}/model.h5")
    
if __name__ == '__main__':
    main()
    
