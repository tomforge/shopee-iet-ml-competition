import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

model1 = load_model("finalEnsemble/inception_v3_hinge_83.h5")
model2 = load_model("finalEnsemble/inception_v4_hinge_84.h5")
model3 = load_model("finalEnsemble/inception_v3_yuj_hinge_84.h5")
model4 = load_model("finalEnsemble/xception_unfreezeall_839.h5")

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
results4 = model4.predict_generator(prediction_generator, verbose=True)

ensemble = np.argmax((results1 + results2 + results3 + results4)/4, axis=1)
# Extract image ids from the filenames (which are in order)
ids = [int(f[10:-4]) for f in prediction_generator.filenames]

# Create the csv for submission
sub = pd.DataFrame({'id':ids, 'category':ensemble})
sub = sub.sort_values(by=['id'])
# Place id column first
sub = sub[['id', 'category']]
sub.to_csv('4ensemble1.csv', index=False)
