import numpy as np
import keras
from keras.applications import inception_resnet_v2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.layers import GaussianNoise, Input


TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
NUM_TRAIN, NUM_TEST = 34398, 3813
# Use ImageNet image sizes
IMG_WIDTH, IMG_HEIGHT = 256, 256
FC_SIZE = 1024
NUM_EPOCHS = 25
BATCH_SIZE = 27
NB_CLASSES = 18
train_datagen = ImageDataGenerator(
		rescale = 1./255,
		horizontal_flip = True,
		fill_mode = "nearest",
		zoom_range = 0.3,
		width_shift_range = 0.3,
		height_shift_range=0.3,
		rotation_range=40)

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
checkpoint = ModelCheckpoint("inception_v4_hinge.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')

trained_model = load_model("inception_v4_finetune_84.h5")
for layer in trained_model.layers:
  layer.trainable = True
# Fine tune using SGD
trained_model.compile(optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_hinge', metrics=['accuracy'])

trained_model.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [checkpoint, early])
