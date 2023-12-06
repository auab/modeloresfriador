import numpy as np


def solve_one_dimensional_diffusion(L, d_x, k, west_bound=None, east_bound=None,
                                    area=1, q_dot=0, h = 0, perimeter = 0, T_conv=0):
    if east_bound is None:
        east_bound = {"condition": "dirichlet", "value": 0}
    if west_bound is None:
        west_bound = {"condition": "dirichlet", "value": 0}


    n_x = int(L/d_x)
    thermal_resist = k*area/d_x
    vol_heat_source = q_dot*area*d_x
    convection_resist = h*perimeter*d_x
    A_matrix = np.zeros((n_x, n_x))
    b_vector = np.zeros(n_x)
    for i in range(n_x):
        # Left boundary
        if i == 0:
            A_matrix[i, i+1] = -thermal_resist
            if west_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist
                b_vector[i] = 2 * thermal_resist * west_bound['value'] + vol_heat_source + \
                              convection_resist * T_conv
            elif west_bound['contidion'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist
                b_vector[i] = vol_heat_source + convection_resist * T_conv
        # Right boundary
        elif i == n_x-1:
            A_matrix[i, i-1] = -thermal_resist
            if east_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist
                b_vector[i] = 2 * thermal_resist * east_bound['value'] + vol_heat_source + \
                              convection_resist*T_conv
            elif east_bound['condition'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist
                b_vector[i] = vol_heat_source + convection_resist * T_conv
        # Not a boundary
        else:
            A_matrix[i, i-1] = -thermal_resist
            A_matrix[i, i] = 2 * thermal_resist + convection_resist
            A_matrix[i, i+1] = -thermal_resist
            b_vector[i] = vol_heat_source + convection_resist*T_conv
    return np.linalg.solve(A_matrix, b_vector)


