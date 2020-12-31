# tensorflow/keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, InputLayer, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import RepeatVector, Permute, Input, Reshape, concatenate
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import backend
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


def get_modelEncDecIResNet():
  embed_input = Input(shape=(1000,))

  # encoder part
  ##### test with 512x512 images
  #encoder_input = Input(shape=(512, 512, 1,))
  #encoder_output = Conv2D(32, (3,3), activation='relu', padding='same',strides=2)(encoder_input)
  #encoder_output = Conv2D(64, (3,3), activation='relu', padding='same',strides=2)(encoder_output)
  ######
  encoder_input = Input(shape=(256, 256, 1,))
  encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
  encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
  encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
  # fusion part
  fusion_output = RepeatVector(32 * 32)(embed_input) 
  fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
  fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
  fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 
  # decoder part
  decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
  return model


def get_modelEncDecIResNetColorClasses():
  #buffer_input = Input(shape=(256,256,1,))
  embed_input = Input(shape=(1000,))

  # encoder part
  encoder_input = Input(shape=(256, 256, 1,))
  encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
  encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
  encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
  encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
  # fusion part
  fusion_output = RepeatVector(32 * 32)(embed_input) 
  fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
  fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
  # --fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)  # original
  fusion_output = Conv2D(512, (1, 1), activation='relu', padding='same')(fusion_output)
  # decoder part
  decoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(fusion_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(decoder_output)
  #decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(decoder_output) # additional layer: higher accuracy
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  #decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(decoder_output) # additional layer: higher accuracy
  decoder_output = Conv2D(394, (3, 3), activation='softmax', padding='same')(decoder_output)  # 394 color classes
  model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
  return model


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(" For training batch {}, loss is {:9.7f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print(" For validation batch {}, loss is {:9.7f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:9.7f} "
            "and mean absolute error is {:9.7f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

class CustomTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)