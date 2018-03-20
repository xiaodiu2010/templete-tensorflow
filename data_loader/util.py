import os
import sys
import glob
import numpy as np
import cv2

def get_files(folder):
    """
    Given path of the folder, returns list of files in it
    :param folder:
    :return:
    """
    print(folder)
    filenames = [file for file in glob.glob(folder + "*/*.png")]
    return filenames


def get_labels(folder, filenames):
    labelnames = [folder + '/'.join(file.split('/')[-2:]) for file in filenames]
    return labelnames


def _read_py_function(filename, label):
    image_decoded = cv2.imread(filename.decode('utf-8'))
    image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    mask = np.round(cv2.imread(label.decode('utf-8'), 0)/50)
    return image_decoded, mask.astype(np.uint8)