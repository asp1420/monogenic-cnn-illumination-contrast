#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya", " Sebastian Salazar-Colores", "Abraham Sanchez", "Sebastian Xambò", "Ulises Cortes" ]
__copyright__ = "Copyright 2019, Gobierno de Jalisco"
__credits__ = ["E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"



import sys
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Conv2D, MaxPooling2D
import os,time,tensorflow as tf,keras,h5py as h5py,cv2,pandas as pd,time,numpy as np,shutil,csv
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from hdf5_utilities import load_HDF5, save_images_and_csv_from_data,save_HDF5
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,History
from keras.utils import plot_model,multi_gpu_model
from sklearn.metrics import confusion_matrix
from monogenic_functions import comp_ph_ori_fr_ones_rgb_list
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,Sequential
from keras import backend as K,optimizers
from keras.utils import multi_gpu_model
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import timedelta
from shutil import copyfile
from PIL import Image
from glob import glob
import pandas as pd
from keras import applications

#---------------------------Measure Time
start_time = time.monotonic()
succesful_proc= False
#-----------------Input-data-----------------------
script_name = sys.argv[0]
job_id = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
lr = float(sys.argv[4])
main_path='./'
dest_path =main_path+'job_out/'
data_folder=main_path+'data/'
#---------------Other inputs --------------------------
gpus = 1
img_size = 32
zoom_range = 0
horizontal_flip = False
vertical_flip = False
rotation_range = 0
shift = 0
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False
s= 1
mw= 3
multi= 1
sig= 0.3
#----------------------Out put location 
#create_folders
label1 = job_id

for label in [label1]:
        if not os.path.exists(os.path.join(dest_path,label)):
                os.makedirs(os.path.join(dest_path,label),0o775)
outputs=os.path.join(dest_path,label)
filename =outputs+'/{job_id}'.format(job_id=job_id)
copyfile(main_path+script_name, filename+'_'+script_name)


#----------------------- Print Area ----------------------

print ('Python Version:', sys.version)
print('Tensorflow Version:', tf.__version__)
print('Keras Version:', keras.__version__)
print('K.image_data_format():', K.image_data_format())
print('job_id', job_id)
print('script_name', script_name)
print('batch_size', batch_size)
print('epochs', epochs)
print('img_size', img_size)
print('learning_rate', lr)
print('Architecture','see summary')
print('zomm_range', zoom_range)
print('rotation_range', rotation_range)
print('horizontal_flip:', horizontal_flip)
print('vertical_flip:', vertical_flip)
print('Shift',shift )
print('K.image_data_format():', K.image_data_format())
#-----------------------------------------------------------

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
    epoch (int): The number of epochs

    # Returns
    lr (float32): learning rate
    """
    
    
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs, num_filters=16,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
                    strides (int): Conv2D square stride dimensions
                            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
                    conv_first (bool): conv-bn-activation (True) or
                                bn-activation-conv (False)

        # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


#-------------------


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
    input_shape (tensor): shape of input image tensor
    depth (int): number of core convolutional layers
    num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
    model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
    num_filters=num_filters_in,
    conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

                 # bottleneck residual unit
            y = resnet_layer(inputs=x,
                            num_filters=num_filters_in,
                            kernel_size=1,
                            strides=strides,
                            activation=activation,
                            batch_normalization=batch_normalization,
                            conv_first=False)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters_in,
                                conv_first=False)
            y = resnet_layer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
            if res_block == 0:
            # linear projection residual shortcut connection to match
            # changed dims
                x = resnet_layer(inputs=x,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
#----------


#------------------------- Load Data

for h5_name in os.listdir(data_folder):
    print('Names',h5_name)
    (x,y)=load_HDF5(data_folder+h5_name)
    #---------------- Compute Monogenic
    print('Computing Monogenic')
    x=comp_ph_ori_fr_ones_rgb_list(x, ss=s,  minWaveLength=mw, mult=multi, sigmaOnf=sig)
    save_HDF5(x,y,dest_path+h5_name)
    x=np.array(x)

    x_train=x[0:36000]
    y_train=y[0:36000]
    x_val=x[36000:48000]
    y_val=y[36000:48000]
    x_test=x[48000:60000]
    y_test=y[48000:60000]


    train_samples=len(x_train)
    val_samples=len(x_val)
    test_samples=len(x_test)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #----------------------- Data Normalization
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_val /= 255
    x_test /= 255
    #--------------------- Checks---------------------------
    if K.image_data_format() != "channels_last":
        K.set_image_data_format("channels_last")


    # -----------------  MODEL  ----------------------
    n = 3
    version = 1
    depth = n * 6 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)

    input_shape = (img_size, img_size, 6)
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
     
    model = resnet_v2(input_shape=input_shape, depth=depth)
    print(model.summary())


    #-------------------- Optimizer---------------------------

    opt = keras.optimizers.Adam(lr= lr_schedule(0))
    #------------------------Model_ Compile--------------------------------------
    # model = multi_gpu_model(model, gpus=gpus) # for multi-gpu power 9
    model.compile(optimizer= opt,loss='categorical_crossentropy',metrics=['accuracy'])
    #-------------------------------- Define checkpoint and History-----------

    checkpoint = ModelCheckpoint(filename+'_'+h5_name, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=False, mode='auto', period=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=0.5e-6)
    callbacks_list = [checkpoint, History(),lr_scheduler,lr_reducer]


    # -----------------------------Training--------------------------------------------
    hist =  model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_val, y_val),
                              callbacks=callbacks_list,
                              verbose=1)

    # --------------------------Logs of loss and acc---------------------------------

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(epochs)
    max_acc_val = max(val_acc)
    min_loss_val =min(val_loss)
    print('best_acc',max(val_acc), 'min loss', min_loss_val)
    #------------------------------------------------------------Plot Figures
    # Plot model accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename+h5_name+'_plot_acc.png', bbox_inches='tight')
    plt.close('all')
    # Plot model loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename+h5_name+'_plot_loss.png', bbox_inches='tight')
    plt.close('all')

    #-------------------inference zone

    #Delete existing last model 
    del model  
    model = load_model(filename+'_'+h5_name)

    print ('Evaluation Best Model')
    test_loss, test_acc=model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1, sample_weight=None, steps=None)
    print('Test_acc',test_acc)

    #------------report result CSV
    row_n=[str(job_id),script_name,str(round(max_acc_val,5)),str(round(min_loss_val,5)), str(round(test_acc,5)),str(round(test_loss,5)),str(lr),str(s),str(mw),str(multi),str(sig),h5_name,time.strftime("%h/%d/%m/%Y")]

    with open(dest_path+'cifar_monogenic_haze_16_may'+'.csv', 'a') as f:
            writer = csv.writer(f, delimiter='\t',lineterminator='\n')
            writer.writerow(row_n)
            f.close()

end_time = time.monotonic()
print(end_time)
print('Time_processing:',timedelta(seconds=end_time - start_time), 'Date', time.strftime("%d/%m/%Y"))

