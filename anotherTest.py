from __future__ import division
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def analyze(y_pred):
  num_files = [338, 265, 192, 191, 197, 211, 310, 318, 144, 228, 298, 291, 182, 161, 168, 129, 81, 109]
  count = 0
  metrics = []
  bc = np.bincount(y_pred)
  total_correct = 0
  for i,num in enumerate(num_files):
    images = y_pred[count:count+num]
    pred_dist = np.bincount(images)
    # We don't want to count the correct label
    pred_dist[i] = 0
    most_freq_wrong = np.argmax(pred_dist)
    num_most_freq_wrong = pred_dist.max()

    true_pos = np.sum(images == i)
    false_pos = bc[i] - true_pos
    metrics = metrics + [(i, true_pos, false_pos, most_freq_wrong, num_most_freq_wrong, num)]
    count = count + num
    total_correct = total_correct + true_pos
  return metrics
def printModelMetric(metrics, model):
  print("-----" + model + "-----")
  for metric in metrics:
    printMetric(metric)
  print("\n")
def printMetric(metric):
  label, tp, fp, mfreqwrong, num_mfw, num = metric
  print("Label " + str(label) + ": Num images: " + str(num) + "; True pos: " + str(tp) + "; False pos: " + str(fp))
  print(">>> Rate truepos: " + str(tp / num) + "; Rate falsepos: " + str(fp / (tp + fp)))
  print(">>> Most frequently predicted as label " + str(mfreqwrong) + " with " + str(num_mfw) + " images.")

print("Loading models...")
model1 = load_model("inception_v4_finetune_84.h5")
model2 = load_model("inception_v3_finetune_828.h5")
womenClass = load_model("womenClass_inception_v4_finetune_749.h5")
print("Models loaded...")
TEST_DIR = 'data/test'
IMG_WIDTH, IMG_HEIGHT = 256, 256
NUM_PREDICT = 3813

# Need to apply the same data preprocessing
test_datagen = ImageDataGenerator(
		rescale = 1./255)

# Set no shuffle so the predictions follow file order
test_gen = test_datagen.flow_from_directory(
		TEST_DIR,
		target_size = (IMG_HEIGHT, IMG_WIDTH),
		class_mode = "categorical",
		shuffle = False)

results1 = model1.predict_generator(test_gen, verbose=True)
results2 = model2.predict_generator(test_gen, verbose=True)
womenRes = womenClass.predict_generator(test_gen, verbose=True)

pred_labels1 = np.argmax(results1, axis=1)
pred_labels2 = np.argmax(results2, axis=1)

ensemble12 = np.argmax((results1 + results2)/2, axis=1)
womenPreds = np.copy(ensemble12)
womenCat = [5, 7, 9, 11, 13, 15, 16]
womenMask = np.full_like(womenPreds, False, dtype=bool)
for i,v in enumerate(womenCat):
  # Get all the rows predicted as women tops
  womenMask = womenMask | (womenPreds == c)
  # Update the women classifier labels to the correct ones
  womenRes[womenRes == i] = v

# Use women classfier values for women tops
womenPreds[womenMask] = womenRes[womenMask]

printModelMetric(analyze(pred_labels1), "inception_v4_84")
printModelMetric(analyze(pred_labels2), "inception_v3_828") 
printModelMetric(analyze(ensemble12), "inception_v4_84 + inception_v3_828 ensemble")
printModelMetric(analyze(womenPreds), "Ensemble + women classifier")
