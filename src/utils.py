import os
import shutil
import numpy as np
from PIL import Image as pil_image
PROJ_DIR = os.getcwd()
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
# Use ImageNet image sizes
IMG_WIDTH, IMG_HEIGHT = 256, 256
FC_SIZE = 1024

def splitDataset(data_dir, train_dir="", test_dir="", test_percentage=0.1):
    """ Tidies up directory names by making them absolute paths, appending
    PROJ_DIR if necesary, before calling splitDatasetAbsolutePaths()
    """
    # If target paths aren't provided, use the default one
    if train_dir == "":
        train_dir = TRAIN_DIR
    if test_dir == "":
        test_dir = TEST_DIR

    data_dir = os.path.join(PROJ_DIR, data_dir)
    train_dir = os.path.join(PROJ_DIR, train_dir)
    test_dir = os.path.join(PROJ_DIR, test_dir)
    splitDatasetAbsolutePaths(data_dir, train_dir, test_dir, test_percentage)

def splitDatasetAbsolutePaths(data_dir, train_dir, test_dir, test_percentage):
    """ Given a dataset directory with category-labelled
    subfolders, split the data into the given training and
    validation directories according to the given percentage.
    (All paths must be absolute)
    Args:
        data_dir: Path of dataset directory
        train_dir: Path of directory to place training data
        test_dir: Path of directory to place validation data
        test_percentage: Percentage of data files to place in validation
    """
    if not os.path.exists(data_dir):
        print("Data directory does not exist!")
        return 0
    elif os.path.exists(train_dir) or os.path.exists(test_dir):
        print("Target directories already exist!")
        return 0
    else:
        os.makedirs(train_dir)
        print("Created directory " + train_dir)
        os.makedirs(test_dir)
        print("Created directory " + test_dir)

    total_num_train = 0
    total_num_test = 0
    for dir, subdirs, files in os.walk(data_dir):
        # Skip root directory
        if(dir == data_dir):
            continue
        label = os.path.basename(dir)
        train_labelled_dir = os.path.join(train_dir, label)
        test_labelled_dir = os.path.join(test_dir, label)
        # Create the category directories
        os.mkdir(train_labelled_dir)
        os.mkdir(test_labelled_dir)

        np.random.shuffle(files)
        # Test images <- files[0] to files[num_test - 1]
        num_test = int(len(files) * test_percentage)
        # Train images <- files[num_test] to files[end]
        num_train = len(files) - num_test
        for file in files[0:num_test]:
            # Copy files into test directory
            file_path = os.path.join(dir, file)
            shutil.copy(file_path, os.path.join(test_labelled_dir, file))
        for file in files[num_test:]:
            # Copy files into train directory
            file_path = os.path.join(dir, file)
            shutil.copy(file_path, os.path.join(train_labelled_dir, file))

        print(label + ": " + str(num_train) + " training files; "
            + str(num_test) + " testing files.")
        total_num_train += num_train
        total_num_test += num_test
    print("Processed " + str(total_num_train) + " training files.")
    print("Processed " + str(total_num_test) + " validation files.")

def removeCorruptedImages(data_dir=""):
    if data_dir == "":
        data_dir = os.path.join(PROJ_DIR, "data")
    if not os.path.exists(data_dir):
        print("Data directory does not exist!")
        return 0

    for dir, subdirs, files in os.walk(data_dir):
        # Skip root directory
        if(dir == data_dir):
            continue
        for file in files:
            file_path = os.path.join(dir, file)
            try:
                img = pil_image.open(file_path)
            except IOError:
                # Remove the corrupted image file
                os.remove(file_path)
                print("Removed corrupted file at " + file_path)
def countImages(train_dir="", test_dir=""):
    if train_dir == "":
        train_dir = os.path.join(PROJ_DIR, TRAIN_DIR)
    if test_dir == "":
        test_dir = os.path.join(PROJ_DIR, TEST_DIR)
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Data directories do not exist!")
        return 0
    num_test = 0
    num_train = 0
    for dir, _, files in os.walk(train_dir):
        num_train += len(files)
    for dir, _, files in os.walk(test_dir):
        num_test += len(files)
    print("Counted " + str(num_train) + " training files.")
    print("Counted " + str(num_test) + " validation files.")
