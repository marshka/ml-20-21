import os
import pickle
import urllib.request as http
from zipfile import ZipFile

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import save_model, load_model


def load_cifar10(num_classes=3):
    """
    Downloads CIFAR-10 dataset, which already contains a training and test set,
    and return the first `num_classes` classes.
    Example of usage:

    >>> (x_train, y_train), (x_test, y_test) = load_cifar10()

    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered data.
    """
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

    fil_train = tf.where(y_train_all[:, 0] < num_classes)[:, 0]
    fil_test = tf.where(y_test_all[:, 0] < num_classes)[:, 0]

    y_train = y_train_all[fil_train]
    y_test = y_test_all[fil_test]

    x_train = x_train_all[fil_train]
    x_test = x_test_all[fil_test]

    return (x_train, y_train), (x_test, y_test)


def load_rps(download=False, path='rps'):
    """
    Downloads the rps dataset and returns the training and test sets.
    Example of usage:

    >>> (x_train, y_train), (x_test, y_test) = load_rps()

    :param download: bool, default is False but for the first call should be True.
    :param path: str, subdirectory in which the images should be downloaded, default is 'rps'.
    :return: the images and labels split into training and validation sets.
    """
    url = 'https://drive.switch.ch/index.php/s/xjXhuYDUzoZvL02/download'
    classes = ('rock', 'paper', 'scissors')
    base_dir = os.path.dirname(os.path.realpath(__file__))
    rps_dir = os.path.join(base_dir, path)
    filename = os.path.join(rps_dir, 'data.zip')
    if not os.path.exists(rps_dir) and not download:
        raise ValueError("Dataset not in the path. You should call this function with `download=True` the first time.")
    if download:
        os.makedirs(rps_dir, exist_ok=True)
        print("Downloading rps images (may take a couple of minutes)")
        path, msg = http.urlretrieve(url, filename)
        with ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(rps_dir)
        os.remove(filename)
    train_dir, test_dir = os.path.join(rps_dir, 'train'), os.path.join(rps_dir, 'test')
    print("Loading training set...")
    x_train, y_train = load_images_with_label(train_dir, classes)
    print("Loaded %d images for training" % len(y_train))
    print("Loading test set...")
    x_test, y_test = load_images_with_label(test_dir, classes)
    print("Loaded %d images for testing" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def make_dataset(imgs, labels, label_map, img_size, rgb=True, keepdim=True, shuffle=True):
    x = []
    y = []
    n_classes = len(list(label_map.keys()))
    for im, l in zip(imgs, labels):
        # preprocess img
        x_i = im.resize(img_size)
        if not rgb:
            x_i = x_i.convert('L')
        x_i = np.asarray(x_i)
        if not keepdim:
            x_i = x_i.reshape(-1)
        
        # encode label
        y_i = np.zeros(n_classes)
        y_i[label_map[l]] = 1.
        
        x.append(x_i)
        y.append(y_i)
    x, y = np.array(x).astype('float32'), np.array(y)
    if shuffle:
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        x, y = x[idxs], y[idxs]
    return x, y


def load_images(path):
    img_files = os.listdir(path)
    imgs, labels = [], []
    for i in img_files:
        if i.endswith('.jpg'):
            # load the image (here you might want to resize the img to save memory)
            imgs.append(Image.open(os.path.join(path, i)).copy())
    return imgs


def load_images_with_label(path, classes):
    imgs, labels = [], []
    for c in classes:
        # iterate over all the files in the folder
        c_imgs = load_images(os.path.join(path, c))
        imgs.extend(c_imgs)
        labels.extend([c] * len(c_imgs))
    return imgs, labels


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model.h5')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = load_model(filename)
    return model


def save_vgg16(model, filename='nn_task2.pkl', additional_args=()):
    """
    Optimize task2 model by only saving the layers after vgg16. This function
    assumes that you only added Flatten and Dense layers. If it is not the case,
    you should include into `additional_args` other layers' attributes you
    need.

    :param filename: string, path to the file in which to store the model.
    :param additional_args: tuple or list, additional layers' attributes to be 
    saved. Default are ['units', 'activation', 'use_bias']
    :return: the path of the saved model.
    """
    filename = filename if filename.endswith('.pkl') else (filename + '.pkl')
    args = ['units', 'activation', 'use_bias', *additional_args]
    layers = []
    for l in model.layers[1:]:
        layer = dict()
        layer['class'] = l.__class__.__name__
        if l.weights:
            layer['weights'] = l.get_weights()
            layer['kwargs'] = {k: v for k, v in vars(l).items() if k in args}
        layers.append(layer)

    with open(filename, 'wb') as fp:
        pickle.dump(layers, fp)
    
    return os.path.abspath(filename)
