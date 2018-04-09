import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image

def get_files(folders, suffix='.png'):
    """
    Given path of the folder, returns list of files in it
    :param folder:
    :return:
    """
    filenames = []
    folders = glob.glob(folders + "*")
    print(folders)
    for folder in folders:
        filenames += glob.glob(folder + "*/*" + suffix)
    return filenames


def get_labels(folder, filenames):
    labelnames = [folder + '/'.join(file.split('/')[-2:]) for file in filenames]
    return labelnames


def _read_py_function(filename, label, suffix='.png'):
    if suffix==".png":
        image_decoded = cv2.imread(filename.decode('utf-8'), 0)
        image_decoded = image_decoded[:, :, np.newaxis]
        mask = cv2.imread(label.decode('utf-8'), 0)

    else:
        image_decoded = np.array(Image.open(filename))
        image_decoded = image_decoded[:, :, np.newaxis]
        image_decoded = cv2.cvtColor(np.array(image_decoded), cv2.COLOR_GRAY2BGR)
        mask = Image.open(label)

    return image_decoded.astype(np.uint8), mask.astype(np.uint8)


def _read_py_function_test(filename):
    image_decoded = cv2.imread(filename.decode('utf-8'), 0)
    image_decoded = image_decoded[:, :, np.newaxis]

    return image_decoded.astype(np.uint8)


def merge_pics(pred_img, true_img):
    n, w, h, c = true_img.shape
    if pred_img.shape[1] != w:
        print("error")
    temp = np.zeros((n, w, 2*h, c))
    temp[:, :, 0:h, :] = true_img
    temp[:, :, h:, :] = pred_img
    return temp


def split_train_valid(folder):
    folders = glob.glob(folder + "*")
    print(folders)
    num_folders = len(folders)
    index = np.random.permutation(np.arange(num_folders))
    folders = np.array(folders)
    train_folders = folders[index[:int(0.9 * num_folders)]]
    valid_folders = folders[index[int(0.9 * num_folders):]]
    return train_folders, valid_folders