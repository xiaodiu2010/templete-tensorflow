import os, sys
sys.path.append('./')
sys.path.append(os.path.dirname(__file__))
print(sys.path)
import tensorflow as tf
#from augmentor import Augmentation
from augmentor2 import Augmentation
from util import *
import util

class DataGenerator(object):
    def __init__(self, config):
        """
        config:
        data_paralle_size
        buffer:
        batch_size:
        """
        self.config = config
        self.train_filenames = get_files(config.train_file_path)
        self.train_labels = get_labels(config.train_label_path, self.train_filenames)
        self.test_filenames = get_files(config.test_file_path)
        self.test_labels = get_labels(config.test_label_path, self.test_filenames)
        self.dataset = None
        self.test_dateset = None
        self.print_info()
        self.build_dataset()
        self.iterator_train = None
        self.iterator_test = None

    def print_info(self):
        print("The dataset has {} examples".format(len(self.train_filenames)))
        print("Their are {} images, and {} masks".format(len(self.train_filenames), len(self.train_labels)))

    def build_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
        self.test_dateset = tf.data.Dataset.from_tensor_slices((self.test_filenames, self.test_labels))

    def get_train_data(self):
        dataset_train = self.dataset.map(
                        lambda filename, label: tuple(tf.py_func(
                            self.load_data, [filename, label], [tf.uint8, tf.uint8])),
                        num_parallel_calls = self.config.data_parallel_threads)

        dataset_train = dataset_train.prefetch(self.config.batch_size)
        dataset_train = dataset_train.shuffle(self.config.buffer)
        dataset_train = dataset_train.repeat(self.config.epoch)
        dataset_train = dataset_train.map(
                        lambda filename, label: tuple(tf.py_func(
                            Augmentation(self.config), [filename, label], [tf.float32, tf.uint8])),
                        num_parallel_calls=self.config.data_parallel_threads)
        dataset_train = dataset_train.batch(self.config.batch_size)
        dataset_train = dataset_train.prefetch(4)

        self.iterator_train = dataset_train.make_initializable_iterator()
        #self.iterator_train = dataset_train.make_one_shot_iterator()
        X, y = self.iterator_train.get_next()
        return X, y

    def get_test_data(self):
        dataset_test = self.test_dateset.map(
                        lambda filename, label: tuple(tf.py_func(
                            self.load_data, [filename, label], [tf.uint8, tf.uint8])),
                        num_parallel = self.config.data_parallel_threads)
        dataset_test = dataset_test.prefetch(self.config.batch_size)
        dataset_test = dataset_test.repeat(1)
        dataset_test = dataset_test.map(
                        lambda filename, label: tuple(tf.py_func(
                            Augmentation(self.config,is_train=False), [filename, label], [tf.float32, tf.uint8])),
                        num_parallel_calls=self.config.data_parallel_threads)
        dataset_test = dataset_test.batch(self.config.batch_size)
        dataset_test = dataset_test.prefetch(4)

        self.iterator_test = dataset_test.make_one_shot_iterator()
        X_test, y_test = self.iterator_test.get_next()
        return X_test, y_test

    def get_iterator(self):
        if self.config.is_train:
            return self.iterator_train
        else:
            return self.iterator_test

    def load_data(self, filename, label):
        image, mask = util._read_py_function(filename, label)
        return image, mask



