from sys import exit, argv
from pca import pca_train, kpca_train

if len(argv) < 2:
    print('No method selected (PCA/KPCA)')
    exit(1)

mode = argv[1]
if mode != 'pca' and mode != 'kpca':
    print('Input mode must be \'pca\' or \'kpca\'')
    exit(1)

print('Please wait, the network is being trained...')
pca_train() if mode == 'pca' else kpca_train()
print('Training done.')
