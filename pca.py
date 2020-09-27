from os import listdir
from os.path import join, isfile
from eigenvectors import find_eigenvectors
from pathlib import Path
import matplotlib.pyplot as plt
import PIL
import numpy as np
from sklearn import svm


IMAGES_PER_PERSON = 3

training_dataset = 'training_set/'
trained_file = 'trained.txt'
folders = [f for f in listdir(training_dataset)]

print(listdir(training_dataset))

persons = len(folders)
trainings = IMAGES_PER_PERSON * persons

# IMAGE SIZE
width = 480
height = 640
image_area = width * height


def pca_train():
    trained = Path(trained_file)
    if trained.is_file():
        print('Successfully loaded {}'.format(trained_file))
        return np.loadtxt(trained)

    all_faces, last_index = parse_faces()
    # Average face. The second arg (0) means that the mean is calculated
    # using all the values at that index from every variant
    mean = np.mean(all_faces, 0)

    # Restando cara media
    all_faces = [all_faces[k, :] - mean for k in range(all_faces.shape[0])]
    # print(all_faces)
    # show_average_face(np.asarray(all_faces[0]))

    A = np.transpose(all_faces)
    n, m = A.shape
    L = np.dot(all_faces, A)

    L_eigenvectors = find_eigenvectors(A, L)
    C_eigenvectors = np.dot(A, L_eigenvectors)
    for i in range(m):
        C_eigenvectors[:, i] /= np.linalg.norm(C_eigenvectors[:, i])

    show_average_face(C_eigenvectors)

    np.savetxt(trained_file, C_eigenvectors, fmt='%s')
    return C_eigenvectors


def kpca_train():
    all_faces = parse_faces()

    return


def parse_faces():

    # Matrix with one vector of pixels (len = image_area) per image training
    all_faces = np.zeros([trainings, image_area])
    # Matrix with one vector of pixels (len = image_area) per image training
    person = np.zeros([trainings, 1])
    image_index = 0

    for person_index in range(0, persons):
        person_folder_path = training_dataset + 'person{}/'.format(person_index)
        files = [f for f in listdir(person_folder_path)]
        for file in files:
            # Reading image as a matrix of pixels
            current_path = person_folder_path + file
            img = PIL.Image.open(current_path).convert('L')

            # Converting the matrix to a long array of values between 0 and 1
            img_array = (np.asarray(img.getdata()) / 255)

            # Adding the matrix to the array
            all_faces[image_index, :] = img_array
            person[image_index, 0] = person_index
            image_index += 1
        person_index += 1
    return all_faces, person, image_index


def pca_face_input(eigenfaces, input_image):

    all_faces, person, last_index = parse_faces()

    # Test set
    test_set = np.zeros([1, image_area])

    # Adapting new image to match in format with the rest (normalized and with vector shape)
    a = input_image / 255.0
    # Reshape to vector for insertion in 'images'
    test_number = 1
    all_faces[last_index, :] = np.reshape(a, [test_number, image_area])

    # CARA MEDIA
    # Mean for pixel i using 'images' columns
    mean = np.mean(all_faces, 0)

    test_set = [test_set[k, :] - mean for k in range(test_set.shape[0])]

    max = 100
    labels = np.zeros([test_set])

    # Solo las primeras eigenfaces
    B = eigenfaces[:, 0:max]

    # Proyectado
    image_projection = np.dot(all_faces, B)
    image_test_projection = np.dot(test_set, B)

    # SVM training
    clf = svm.LinearSVC()
    clf.fit(image_projection, person.ravel())
    labels = clf.predict(image_test_projection)

    return labels[0]


def show_average_face(av_face):
    resized = np.asarray([f for f in av_face]).reshape(640, 480)*255
    plt.imshow(resized, cmap="gray")
    plt.show()

