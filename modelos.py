import numpy as np


def solve_one_dimensional_diffusion(L, d_x, Ta, Tb, k, area=1, q_dot=0):
    n_x = int(L/d_x)
    A_matrix = np.zeros((n_x, n_x))
    b_vector = np.zeros(n_x)
    for i in range(n_x):
        if i == 0:
            A_matrix[i, i+1] = -k * area / d_x
            A_matrix[i, i] = 3 * k * area / d_x
            b_vector[i] = 2 * k * area * Ta / d_x + q_dot*area*d_x

        elif i == n_x-1:
            A_matrix[i, i-1] = -k * area / d_x
            A_matrix[i, i] = 3 * k * area / d_x
            b_vector[i] = 2 * k * area * Tb / d_x + q_dot*area*d_x

        else:
            A_matrix[i, i-1] = -k * area / d_x
            A_matrix[i, i] = 2 * k * area / d_x
            A_matrix[i, i+1] = -k * area / d_x
            b_vector[i] = q_dot*area*d_x
    return np.linalg.solve(A_matrix, b_vector)


