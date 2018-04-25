import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
# 84.7 % ensemble
#model1 = load_model("inception_v4_finetune_83.h5")
#model2 = load_model("inception_v3_finetune_82.h5")
model1 = load_model("inception_v4_finetune_84.h5")
model2 = load_model("inception_v3_finetune_828.h5")
model3 = load_model("inception_v3_yuj_84.h5")
womenClass = load_model("womenClass_inception_v4_finetune_749.h5")

PREDICT_DIR = "TestImages"
IMG_WIDTH, IMG_HEIGHT = 256, 256
NUM_PREDICT = 16111

# Need to apply the same data preprocessing
prediction_datagen = ImageDataGenerator(
		rescale = 1./255)

# Set no shuffle so the predictions follow file order
prediction_generator = prediction_datagen.flow_from_directory(
		PREDICT_DIR,
		target_size = (IMG_HEIGHT, IMG_WIDTH),
		class_mode = "categorical",
		shuffle = False)

results1 = model1.predict_generator(prediction_generator, verbose=True)
results2 = model2.predict_generator(prediction_generator, verbose=True)
results3 = model3.predict_generator(prediction_generator, verbose=True)

womenRes = womenClass.predict_generator(prediction_generator, verbose=True)

pred = np.argmax((results1 + results2 + results3)/3, axis=1)
womenCat = [5, 7, 9, 11, 13, 15, 16]
womenMask = np.full_like(pred, False, dtype=bool)
for i,v in enumerate(womenCat):
  # Get all the rows predicted as women tops
  womenMask = womenMask | (preds == c)
  # Update the women classifier labels to the correct ones
  womenRes[womenRes == i] = v

# Use women classfier values for women tops
pred[womenMask] = womenRes[womenMask]

# Extract image ids from the filenames (which are in order)
ids = [int(f[10:-4]) for f in prediction_generator.filenames]
# Create the csv for submission
sub = pd.DataFrame({'id':ids, 'category':pred})
sub = sub.sort_values(by=['id'])
# Place id column first
sub = sub[['id', 'category']]
sub.to_csv('womenSpecialsubmission.csv', index=False)
