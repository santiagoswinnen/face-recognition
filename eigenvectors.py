import numpy as np

EPSILON = 1e-2


def find_eigenvectors(A, L):
    last_R = np.zeros(A.shape)
    L_eigenvectors = 1
    eigenvectors_found = False

    while not eigenvectors_found:
        Q, R = gram_schmidt(L)
        L = np.dot(R, Q)
        L_eigenvectors = np.dot(L_eigenvectors, Q)
        eigenvectors_found = found_eigenvectors(last_R, R)
        last_R = R
    return L_eigenvectors


def gram_schmidt(A):
    (m, n) = A.shape
    R = np.zeros(shape=(n, n))
    Q = np.zeros(shape=(m, n))

    for k in range(0, n):
        R[k, k] = np.linalg.norm(A[0:m, k])
        Q[0:m, k] = A[0:m, k] / R[k, k]

        for j in range(k + 1, n):
            R[k, j] = np.dot(np.transpose(Q[0:m, k]), A[0:m, j])
            A[0:m, j] = A[0:m, j] - np.dot(Q[0:m, k], R[k, j])

    return Q, R


def found_eigenvectors(old_R, new_R):
    for i in range(0, old_R.shape[0]):
        if abs(old_R[i][i] - new_R[i][i]) > EPSILON:
            return False
    return True
