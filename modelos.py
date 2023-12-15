import numpy as np


def solve_1d_diffusion_convection(material_props, mesh, west_bound,
                                               east_bound, external_convection=None, volumetric_source=None,
                                               velocity_field=None):
    velocity_field, external_convection, volumetric_source = initialize_1d_solver(velocity_field,
                                                                                               external_convection,
                                                                                               volumetric_source)
    n_x = int(mesh["L_x"] / mesh["d_x"])

    thermal_resist, vol_heat_source, convection_resist, F_e, F_w = define_1d_equation_components(n_x, mesh,
                                                                                                              material_props,
                                                                                                              volumetric_source,
                                                                                                              velocity_field,
                                                                                                              external_convection)

    A, b = np.zeros((n_x, n_x)), np.zeros(n_x)

    for i in range(n_x):
        S_u = vol_heat_source + convection_resist * external_convection['T_conv']
        S_p = -convection_resist
        a_w = thermal_resist + np.max([F_w, 0])
        a_e = thermal_resist + np.max([-F_e, 0])
        if i == 0:
            a_w = 0
            if west_bound['condition'] == 'dirichlet':
                S_p += -(2 * thermal_resist + np.max([F_w, 0]))
                S_u += (2 * thermal_resist + np.max([F_w, 0])) * west_bound['value']
        elif i == n_x - 1:
            a_e = 0
            if east_bound['condition'] == 'dirichlet':
                S_p += -(2 * thermal_resist + np.max([-F_e, 0]))
                S_u += (2 * thermal_resist + np.max([-F_e, 0])) * east_bound['value']

        if i != 0:
            A[i, i - 1] = -a_w
        if i != (n_x - 1):
            A[i, i + 1] = -a_e

        A[i][i] = a_w + a_e - S_p + (F_e - F_w)
        b[i] = S_u

    return np.linalg.solve(A, b)


def initialize_1d_solver(velocity_field, external_convection, volumetric_source):
    if velocity_field is None:
        velocity_field = {"u": 0}
    if external_convection is None:
        external_convection = {'perimeter': 0, 'h': 0, 'T_conv': 0}
    if volumetric_source is None:
        volumetric_source = {'q_dot': 0}
    return velocity_field, external_convection, volumetric_source


def define_1d_equation_components(n_x, mesh, material_props, volumetric_source, velocity_field, external_convection):
    thermal_resist = material_props["k"] * mesh["area"] / mesh["d_x"]
    vol_heat_source = volumetric_source["q_dot"] * mesh["area"] * mesh["d_x"]
    convection_resist = external_convection['h'] * external_convection['perimeter'] * mesh["d_x"]
    F_e = material_props["rho"] * velocity_field["u"] * mesh["area"]
    F_w = material_props["rho"] * velocity_field["u"] * mesh["area"]
    return thermal_resist, vol_heat_source, convection_resist, F_e, F_w










def solve_2d_diffusion_convection(material_props, mesh, west_bound, north_bound, south_bound,
                                               east_bound, external_convection=None, volumetric_source=None,
                                               velocity_field=None):
    velocity_field, external_convection, volumetric_source = initialize_2d_solver(velocity_field,
                                                                                               external_convection,
                                                                                               volumetric_source)
    n_x = int(round(mesh["L_x"] / mesh["d_x"]))
    n_y = int(round(mesh["L_y"] / mesh["d_x"]))

    thermal_resist, vol_heat_source, convection_resist, F_e, F_w, F_n, F_s = define_2d_equation_components(mesh,
                                                                                                              material_props,
                                                                                                              volumetric_source,
                                                                                                              velocity_field,
                                                                                                              external_convection)

    A, b = np.zeros((n_x*n_y, n_x*n_y)), np.zeros(n_x*n_y)

    for i in range(n_x):
        for j in range(n_y):
            S_u = vol_heat_source + convection_resist * external_convection['T_conv']
            S_p = -convection_resist
            a_w = thermal_resist + np.max([F_w, 0])
            a_e = thermal_resist + np.max([-F_e, 0])
            a_s = thermal_resist + np.max([F_s, 0])
            a_n = thermal_resist + np.max([-F_n, 0])
            if i == 0:
                a_w = 0
                if west_bound['condition'] == 'dirichlet':
                    S_p += -(2 * thermal_resist + np.max([F_w, 0]))
                    S_u += (2 * thermal_resist + np.max([F_w, 0])) * west_bound['value']
                elif (west_bound['condition'] == 'neumann') & (west_bound['value'] != 0):
                    S_u += west_bound['value']*mesh['area']
            if i == n_x - 1:
                a_e = 0
                if east_bound['condition'] == 'dirichlet':
                    S_p += -(2 * thermal_resist + np.max([-F_e, 0]))
                    S_u += (2 * thermal_resist + np.max([-F_e, 0])) * east_bound['value']
                elif (east_bound['condition'] == 'neumann') & (east_bound['value'] != 0):
                    S_u += east_bound['value'] * mesh['area']
            if j == 0:
                a_s = 0
                if south_bound['condition'] == 'dirichlet':
                    S_p += -(2 * thermal_resist + np.max([F_s, 0]))
                    S_u += (2 * thermal_resist + np.max([F_s, 0])) * south_bound['value']
                elif (south_bound['condition'] == 'neumann') & (south_bound['value'] != 0):
                    S_u += south_bound['value']*mesh['area']

            if j == n_y - 1:
                a_n = 0
                if north_bound['condition'] == 'dirichlet':
                    S_p += -(2 * thermal_resist + np.max([-F_n, 0]))
                    S_u += (2 * thermal_resist + np.max([-F_n, 0])) * north_bound['value']
                elif (north_bound['condition'] == 'neumann') & (north_bound['value'] != 0):
                    S_u += north_bound['value']*mesh['area']

            if i != 0:
                A[i + j*n_x, (i + j*n_x) - 1] = -a_w
            if i != (n_x - 1):
                A[i + j*n_x, (i + j*n_x) + 1] = -a_e
            if j != 0:
                A[i + j*n_x, (i + (j-1)*n_x)] = -a_s
            if j != (n_y-1):
                A[i + j * n_x, (i + (j + 1) * n_x)] = -a_n

            A[i + j * n_x][i + j * n_x] = a_w + a_e + a_n + a_s - S_p + (F_e - F_w) + (F_n - F_s)
            b[i + j * n_x] = S_u
    return np.linalg.solve(A, b)


def initialize_2d_solver(velocity_field, external_convection, volumetric_source):
    if velocity_field is None:
        velocity_field = {"u": 0, "v": 0}
    if external_convection is None:
        external_convection = {'perimeter': 0, 'h': 0, 'T_conv': 0}
    if volumetric_source is None:
        volumetric_source = {'q_dot': 0}
    return velocity_field, external_convection, volumetric_source


def define_2d_equation_components(mesh, material_props, volumetric_source, velocity_field, external_convection):
    thermal_resist = material_props["k"] * mesh["area"] / mesh["d_x"]
    vol_heat_source = volumetric_source["q_dot"] * mesh["area"] * mesh["d_x"]
    convection_resist = external_convection['h'] * external_convection['perimeter'] * mesh["d_x"]
    F_e = material_props["rho"] * velocity_field["u"] * mesh["area"]
    F_w = material_props["rho"] * velocity_field["u"] * mesh["area"]
    F_n = material_props["rho"] * velocity_field["v"] * mesh["area"]
    F_s = material_props["rho"] * velocity_field["v"] * mesh["area"]
    return thermal_resist, vol_heat_source, convection_resist, F_e, F_w, F_n, F_s
