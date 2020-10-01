# -*- coding: utf-8 -*-

from pca import kpca_train, pca_train
from utils import build_dict
from cv2_implementation import detect_faces
import sys

if len(sys.argv) < 2:
    print('An input mode is required: \'pca\' or \'kpca\'')
    sys.exit(1)
mode = sys.argv[1]
if mode != 'pca' and mode != 'kpca':
    print('Invalid option. Input mode must be either \'pca\' or \'kpca\'')
    sys.exit(1)

print('Please wait. Training is in progress...')
if mode == 'pca':
    eigenfaces = pca_train()
else:
    eigenfaces = kpca_train()
print('Training ready.')

names = build_dict()
detect_faces(mode, eigenfaces, names)






