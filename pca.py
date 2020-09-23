from os import listdir
from os.path import join, isdir
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

TRAINING_PER_PERSON = 6

training_dataset = 'training_set/'
trained_file = 'trained.txt'

persons = len(directories)
images_per_person = 6
trainings = images_per_person * persons

# IMAGE SIZE
width = 92
height = 112
image_area = width * height


def pca_train():
    trained = Path(trained_file)
    if trained.is_file():
        print('Successfully loaded {}'.format(trained_file))
        return np.loadtxt(trained)

    # Matrix with one vector of pixels (len = image_area) per image training
    images = np.zeros([trainings, image_area])
    # Matrix with one vector of pixels (len = image_area) per image training
    person = np.zeros([trainings, 1])
    image_index = 0

    for i in range(1, images_per_person + 1):
        # Reading image as a matrix of pixels
        pixel_matrix = plt.imread(training_dataset + '/{}'.format(i) + '.pgm')/255.0
        # Adding the matrix to the array
        images[image_index, :] = np.reshape(pixel_matrix, [1, image_area])
        person[image_index, 0] = folder.split('s')[1]
        image_index += 1

    return


def kpca_train():
    return
