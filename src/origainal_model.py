import os
import tensorflow as tf
import numpy as np\
    
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

import pathlib
from keras.utils import to_categorical
from encode_to_seq import Encode2Seq

from datetime import datetime
import os
import glob


def get_batch_files(training_count, test_count):
    # file_root = tf.keras.utils.get_file(
    # 	'simulated_cyp2d6_diplotypes',
    # 	'https://zenodo.org/record/3951095/files/simulated_cyp2d6_diplotypes.tar.gz',
    # 	untar=True
    # )
    
    file_root =  '../data/simulated_cyp2d6_diplotypes'
    
    file_root = pathlib.Path(file_root)
    filenames = []
    for f in file_root.glob("*"):
        filenames.append(f)

    _filenames = np.array([f.name.split('.')[0] for f in filenames])
    batch_names = np.unique(_filenames)
    filenames = np.array([str(f.absolute()) for f in filenames])
    training_batches, test_batches = [], []

    for i, b in enumerate(batch_names):
        if i >= test_count + training_count:
            break
        
        if i < training_count:
            training_batches.append(filenames[_filenames == b])
        else:
            test_batches.append(filenames[_filenames == b])

    return training_batches, test_batches

# def hot_encode_float(y):
# 	classes = []
# 	values = np.unique(y)
# 	for i in range(len(values)):
# 		classes.append(str(i))
# 	encoded_classes = to_categorical(classes)
# 	conversion_dict = dict(zip(values, range(5)))
# 	encoded_y = np.array([encoded_classes[conversion_dict[i]] for i in y])

# 	return encoded_y

def label2onehot(label: np.array) -> np.array:
    thresholds = [0.25, 0.75, 1.25, 1.75]

    categorical_labels = np.digitize(label, thresholds)

    one_hot_encoder = to_categorical(categorical_labels)
    
    return one_hot_encoder

ANNOTATIONS = '../data/gvcf2seq.annotation_embeddings.csv'
EMBEDDINGS = '../data/embeddings.txt'
REF = '../data/ref.seq'

def generate_data(batches):
    for filenames in batches:
        vcf = 0 if 'vcf' == filenames[0].decode('utf-8').split('.')[-1] else 1
        labels = 1 - vcf
        encoding = Encode2Seq(vcf=filenames[vcf].decode('utf-8'), labels=filenames[labels].decode('utf-8'), embedding_file=EMBEDDINGS, annotation_file=ANNOTATIONS, ref_seq=REF)
        y = label2onehot(encoding.y.flatten())
        # print(y)
        for i in range(encoding.X.shape[0]):
            yield encoding.X[i], y[i]
   
# Convolution layers based on final model from paper:
# https://github.com/gregmcinnes/Hubble2D6/blob/master/data/models/hubble2d6_0.json

def cnn_block(layer, filters_num: int, kernel_size: int, strides: int, pool_size: int, pool_strides: int, name_index: int):
    
    if name_index == 1:
        layer = Conv1D(
            filters=filters_num,
            kernel_size=kernel_size,
            strides=strides,
            input_shape=(14868, 13),
            activation='linear',
            name=f'conv_{name_index}'
        )(layer)
    else: 
        layer = Conv1D(
            filters=filters_num,
            kernel_size=kernel_size,
            strides=strides,
            activation='linear',
            name=f'conv_{name_index}'
        )(layer)
    layer = BatchNormalization(name=f'batch_{name_index}')(layer)
    layer = tf.keras.layers.ReLU(name=f'relu_{name_index}')(layer)
    
    # layer = Activation(activation='relu', name=f'activation_{name_index}')(layer)
    layer = MaxPooling1D(pool_size=pool_size, strides=pool_strides, name=f'max_pooling_{name_index}')(layer)
    
    return layer

def build_model(input_shape):
    inputs = Input(shape=input_shape, dtype=float, name="data")

    layer = cnn_block(layer=inputs, filters_num=70, kernel_size=19, strides=5, pool_size=3, pool_strides=3, name_index=1)
    layer = cnn_block(layer=layer, filters_num=46, kernel_size=11, strides=5, pool_size=4, pool_strides=4, name_index=2)
    layer = cnn_block(layer=layer, filters_num=46, kernel_size=7, strides=5, pool_size=4, pool_strides=4, name_index=3)
    
    layer = Flatten(name='flatten')(layer)
    layer = Dense(units=32, activation='relu', name='dense1')(layer)
    layer = Dropout(0.3, name='dropout_4')(layer)
    
    outputs = Dense(units=5, activation='softmax', name='dense2')(layer)
    
    model = Model(inputs=inputs, outputs=outputs, name='cnn_model')
    
    return model


def get_model():
    inputs = Input(shape=(14868, 13), name="data")
    
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
 

def main():
    input_shape= (14868, 13)
    model = get_model()
 
    model.summary()
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam,
                    loss=tf.keras.losses.CategoricalCrossentropy(), 
                    metrics=['accuracy'])	
    
    batch_size = 100
    epochs = 5
    steps_per_epoch = 50000 // batch_size
 
    training_batches, test_batches = get_batch_files(100, 20)
    
    train_dataset = tf.data.Dataset.from_generator(generate_data, args=[training_batches], output_types=(tf.float32, tf.float32), output_shapes=((14868, 13), (5,)))
    test_dataset = tf.data.Dataset.from_generator(generate_data, args=[test_batches], output_types=(tf.float32, tf.float32), output_shapes=((14868, 13), (5,)))
 
    train_dataset = train_dataset.shuffle(500).repeat(count=5).batch(batch_size)
    test_dataset = test_dataset.batch(500)
 
 
 
    now = datetime.now().strftime('%d-%m-%Y_%H-%M')
    
    model_name = os.path.basename(__file__).split('.')[0]
    
    checkpoint_callback = ModelCheckpoint(filepath=f"../model/ModelCheckPoint/{model_name}_{now}/" + "model.{epoch:03d}-{loss:.4f}-{accuracy:.4f}.h5",
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        verbose=1)
    tensorboard_callback = TensorBoard(log_dir=f"../model/TensorBoard/{model_name}_{now}/logs")

    folder_logger_path = f"../model/CSVLogger/{model_name}_{now}"

    if os.path.exists(folder_logger_path) and os.path.isdir(folder_logger_path):
        os.rmdir(folder_logger_path)
    os. makedirs(folder_logger_path)

    csv_logger_callback = CSVLogger(f"{folder_logger_path}/training.log")
    model.fit(train_dataset,
           epochs=epochs, 
           steps_per_epoch=steps_per_epoch,
           callbacks=[checkpoint_callback,
                        tensorboard_callback,
                        csv_logger_callback]
            )
    model.save(f"../model/FinalModel/{model_name}_{now}/model.h5")


if __name__ == '__main__':
    main()