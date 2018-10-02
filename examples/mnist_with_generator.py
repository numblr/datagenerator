import logging
from pprint import pprint

from tempfile import mkstemp
from os.path import basename, splitext
from os import makedirs, fdopen

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
from pandas import DataFrame
from keras.datasets import mnist

from numblr.datagenerator import (LabelEncoder, IntToOneHotEncoder,
        generator_for_files)


logger = logging.getLogger()


MNIST_DIR = 'mnist'
MNIST_PREFIX = 'mnist_'
MNIST_SUFFIX = '.txt'
MNIST_INVENTORY = 'mnist/inventory.csv'


BATCH_SIZE = 128
EPOCHS = 4
NUM_CLASSES = 10


def setupFiles(num_files):
    try:
        makedirs(MNIST_DIR)
    except:
        logger.info("Init: Example data is already setup. Remove '%s' directory to reset.", MNIST_DIR)
        return

    train, _ = mnist.load_data()
    labeled_images = [ sample for sample in zip(train[0], train[1]) ]

    inventory_list = []
    for image, label in labeled_images[:num_files]:
        file_handle, path = mkstemp(prefix=MNIST_PREFIX, suffix=MNIST_SUFFIX, dir=MNIST_DIR, text = True)
        with fdopen(file_handle, 'wb') as file:
            np.savetxt(file, image)
        inventory_list.append({ 'id': splitext(basename(path))[0], 'target': label })

    inventory = DataFrame.from_records(inventory_list)
    inventory.to_csv(MNIST_INVENTORY)

    logger.info("Init: Setup example with %s files.", inventory.shape[0])


def simple_nn_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(), metrics=['accuracy'])

    return model


def encode_image_file(file_handle):
    image_data = np.loadtxt(file_handle).reshape((784,)).astype('float32')
    image_data /= 255

    return image_data


def main(files=12500):
    setupFiles(files)

    data_encoder = encode_image_file
    target_encoders = [LabelEncoder(), IntToOneHotEncoder()]

    generator_data_set = generator_for_files(MNIST_INVENTORY, MNIST_DIR,
            data_encoder, target_encoders,
            id_mapper=lambda id: id + MNIST_SUFFIX,
            id='id', target='target', binary=False)

    train, validation, test = generator_data_set.split(validation=0.2, test=0.2)
    logger.info("Loaded training (%s), validation (%s) and test (%s) data sets",
            train.size, validation.size, test.size)

    model = simple_nn_model()
    model.fit_generator(generator=train.batches(batch_size=BATCH_SIZE),
            steps_per_epoch=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation.batches(batch_size=BATCH_SIZE),
            validation_steps=BATCH_SIZE,
            verbose=1)

    # score = model.evaluate(test.data(), test.targets(), verbose=0)
    score = model.evaluate_generator(test.batches(batch_size=BATCH_SIZE),
            steps=test.size//BATCH_SIZE)

    logger.info('Test loss: %s', score[0])
    logger.info('Test accuracy: %s', score[1])


if __name__ == '__main__':
    import sys

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    main()
