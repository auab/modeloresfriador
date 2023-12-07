import numpy as np


def solve_one_dimensional_diffusion(mesh=None, west_bound=None, east_bound=None, q_dot=0, h=0, perimeter=0,
                                    T_conv=0, material_props=None):
    if east_bound is None:
        east_bound = {"condition": "dirichlet", "value": 0}
    if west_bound is None:
        west_bound = {"condition": "dirichlet", "value": 0}

    n_x = int(mesh["L_x"] / mesh["d_x"])
    thermal_resist = material_props["k"] * mesh["area"] / mesh["d_x"]
    vol_heat_source = q_dot * mesh["area"] * mesh["d_x"]
    convection_resist = h * perimeter * mesh["d_x"]
    A_matrix = np.zeros((n_x, n_x))
    b_vector = np.zeros(n_x)

    for i in range(n_x):
        # Left boundary
        b_vector[i] = vol_heat_source + convection_resist * T_conv
        if i != 0:
            A_matrix[i, i - 1] = -thermal_resist
        if i != (n_x - 1):
            A_matrix[i, i + 1] = -thermal_resist

        if i == 0:
            if west_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist
                b_vector[i] += 2 * thermal_resist * west_bound['value']
            elif west_bound['condition'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist

        # Right boundary
        elif i == n_x - 1:
            if east_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist
                b_vector[i] += 2 * thermal_resist * east_bound['value']
            elif east_bound['condition'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist
        # Not a boundary
        else:
            A_matrix[i, i] = 2 * thermal_resist + convection_resist

    return np.linalg.solve(A_matrix, b_vector)


def solve_one_dimensional_diffusion_convection(mesh=None, west_bound=None, east_bound=None, q_dot=0, h=0, perimeter=0,
                                                T_conv=0, material_props=None,velocity_field=None):
    if east_bound is None:
        east_bound = {"condition": "dirichlet", "value": 0}
    if west_bound is None:
        west_bound = {"condition": "dirichlet", "value": 0}
    if velocity_field is None:
        velocity_field = {"u": 0}

    n_x = int(mesh["L_x"] / mesh["d_x"])
    thermal_resist = material_props["k"] * mesh["area"] / mesh["d_x"]
    vol_heat_source = q_dot * mesh["area"] * mesh["d_x"]
    convection_resist = h * perimeter * mesh["d_x"]
    F_e = material_props["rho"]*velocity_field["u"]*mesh["area"]
    F_w = material_props["rho"] * velocity_field["u"] * mesh["area"]

    A_matrix = np.zeros((n_x, n_x))
    b_vector = np.zeros(n_x)

    for i in range(n_x):
        # Left boundary
        b_vector[i] = vol_heat_source + convection_resist * T_conv
        if i != 0:
            A_matrix[i, i - 1] = -thermal_resist - np.max([F_w, 0])
        if i != (n_x - 1):
            A_matrix[i, i + 1] = -thermal_resist - np.max([-F_e, 0])

        if i == 0:
            if west_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist + np.max([F_w, 0]) + np.max([-F_e, 0])
                b_vector[i] += (2 * thermal_resist + np.max([F_w, 0])) * west_bound['value']
            elif west_bound['condition'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist + np.max([F_w, 0]) + np.max([-F_e, 0])

        # Right boundary
        elif i == n_x - 1:
            if east_bound['condition'] == 'dirichlet':
                A_matrix[i, i] = 3 * thermal_resist + convection_resist + np.max([F_w, 0]) + np.max([-F_e, 0])
                b_vector[i] += (2 * thermal_resist + np.max([-F_e, 0])) * east_bound['value']
            elif east_bound['condition'] == 'neumann':
                A_matrix[i, i] = thermal_resist + convection_resist + np.max([F_w, 0]) + np.max([-F_e, 0])
        # Not a boundary
        else:
            A_matrix[i, i] = 2 * thermal_resist + convection_resist + np.max([F_w, 0]) + np.max([-F_e, 0])
    return np.linalg.solve(A_matrix, b_vector)