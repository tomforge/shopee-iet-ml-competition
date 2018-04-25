import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import numpy as np
import keras
from keras.applications import xception
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
NUM_TRAIN, NUM_TEST = 34398, 3813
# Use ImageNet image sizes
IMG_WIDTH, IMG_HEIGHT = 256, 256
FC_SIZE = 1024
NUM_EPOCHS = 25
BATCH_SIZE = 18
NB_CLASSES = 18
train_datagen = ImageDataGenerator(
		rescale = 1./255,
		horizontal_flip = True,
		fill_mode = "nearest",
		zoom_range = 0.3,
		width_shift_range = 0.3,
		height_shift_range=0.3,
		rotation_range=30)

# For test data, rescaling will do
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
		TRAIN_DIR,
		target_size = (IMG_HEIGHT, IMG_WIDTH),
		batch_size = BATCH_SIZE, 
		class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
		TEST_DIR,
		target_size = (IMG_HEIGHT, IMG_WIDTH),
		class_mode = "categorical")

#model = xception.Xception(include_top=False, weights='imagenet', pooling='avg', input_shape = (IMG_WIDTH,IMG_HEIGHT,3))
# Freeze the layers
#for layer in model.layers:
#	layer.trainable = False
#x = model.output
#x = Dense(FC_SIZE, activation='relu')(x)
#x = Dropout(0.5)(x)
#predictions = Dense(NB_CLASSES, activation='softmax')(x)

#final_model = Model(inputs = model.input, outputs = predictions)

#final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#checkpoint = ModelCheckpoint("xception.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')


#final_model.fit_generator(
#		train_generator,
#		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
#		epochs = NUM_EPOCHS,
#		validation_data = test_generator,
#		callbacks = [checkpoint, early])
final_model = load_model("xception_transfer.h5")
# Unfreeze all
for layer in final_model.layers:
  layer.trainable = True
reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.4, patience=4, verbose=1)
final_model.compile(optimizer = SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

unfreeze_checkpoint = ModelCheckpoint("xception_unfreezeall.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
final_model.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [unfreeze_checkpoint, early, reduceLR])
