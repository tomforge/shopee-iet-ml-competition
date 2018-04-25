import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras.applications import inception_resnet_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.optimizers import SGD
from keras.layers import Dropout, Dense, ELU, BatchNormalization

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
NUM_TRAIN, NUM_TEST = 34398, 3813
# Use ImageNet image sizes
IMG_WIDTH, IMG_HEIGHT = 256, 256
FC_SIZE = 2048
NUM_EPOCHS = 20
BATCH_SIZE = 54
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
checkpoint = ModelCheckpoint("inception_v3_difftop.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, verbose=1, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.4, patience=4, verbose=1)




inception3 = load_model("inception_v3_hinge_83.h5")

for layer in inception3.layers:
	layer.trainable = False

x = inception3.layers[-4].output
x = Dense(FC_SIZE, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(NB_CLASSES, activation='softmax')(x)
new_model = Model(inputs=inception3.input, outputs=pred)
new_model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])

new_model.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [checkpoint, early, reduceLR])





inception4 = load_model("inception_v4_hinge_84.h5")
for layer in inception4.layers:
  layer.trainable = False
x = inception4.layers[-4].output
x = Dense(512)(x)
x = BatchNormalization()(x)
x = ELU()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(NB_CLASSES, activation='softmax')(x)
new_model_2 = Model(inputs=inception4.input, outputs=pred)
checkpoint = ModelCheckpoint("inception_v4_difftop.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
new_model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

new_model_2.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [checkpoint, early, reduceLR])







BATCH_SIZE=18
for layer in new_model.layers:
  layer.trainable = True
checkpoint = ModelCheckpoint("inception_v3_difftop_finetune.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
new_model.compile(optimizer = SGD(lr=0.0008, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [checkpoint, early, reduceLR])






for layer in new_model_2.layers:
  layer.trainable = True
checkpoint = ModelCheckpoint("inception_v4_difftop_finetune.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

new_model_2.compile(optimizer = SGD(lr=0.0008, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
new_model_2.fit_generator(
		train_generator,
		steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
		epochs = NUM_EPOCHS,
		validation_data = test_generator,
		callbacks = [checkpoint, early, reduceLR])
