import numpy as np


def solve_one_dimensional_diffusion(Area, L, d_x, Ta, Tb, k):
    n_x = int(L/d_x)
    A_matrix = np.zeros((n_x,n_x))
    b_vector = np.zeros(n_x)
    for i in range(n_x):
        if i==0:
            A_matrix[i,i+1] = -k * Area / d_x
            A_matrix[i,i] = 3 * k * Area / d_x
            b_vector[i] = 2 * k * Area * Ta / d_x

        elif i==n_x-1:
            A_matrix[i,i-1] = -k * Area / d_x
            A_matrix[i,i] = 3 * k * Area / d_x
            b_vector[i] = 2 * k * Area * Tb / d_x

        else:
            A_matrix[i,i-1] = -k * Area / d_x
            A_matrix[i,i] = 2 * k * Area / d_x
            A_matrix[i,i+1] = -k * Area / d_x
            b_vector[i] = 0
    return np.linalg.solve(A_matrix,b_vector)


