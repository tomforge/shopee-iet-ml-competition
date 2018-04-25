import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

model = load_model("inceptionv4_finetune_84.h5")
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

results = model.predict_generator(prediction_generator, verbose=True)

pred_labels = np.argmax(results, axis=1)
# Extract image ids from the filenames (which are in order)
ids = [int(f[10:-4]) for f in prediction_generator.filenames]
# Create the csv for submission
sub = pd.DataFrame({'id':ids, 'category':pred_labels})
sub = sub.sort_values(by=['id'])
# Place id column first
sub = sub[['id', 'category']]
sub.to_csv('submission.csv', index=False)

