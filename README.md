#Working setup of transfer learning using Keras

# `src.utils` 
Couple of utility function to deal with the data.
`removeCorruptedImages()` will remove images that cannot be opened by PIL.
`splitDataset()` splits the original training dataset into training and validation folders.
Example usage (assuming this is run from project root directory):
```
import src.utils as util
util.splitDataset("TrainingImages")
```
This will take <proj-dir>/TrainingImages and split the data into <proj-dir>/data/train and <proj-dir>/data/test
