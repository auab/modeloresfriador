import numpy as np


def solve_one_dimensional_diffusion_convection(material_props, mesh, west_bound,
                                               east_bound, external_convection=None, volumetric_source=None,
                                               velocity_field=None):
    if velocity_field is None:
        velocity_field = {"u": 0}
    if external_convection is None:
        external_convection = {'perimeter': 0, 'h': 0, 'T_conv': 0}
    if volumetric_source is None:
        volumetric_source = {'q_dot': 0}


    n_x = int(mesh["L_x"] / mesh["d_x"])
    thermal_resist = material_props["k"] * mesh["area"] / mesh["d_x"]
    vol_heat_source = volumetric_source["q_dot"] * mesh["area"] * mesh["d_x"]
    convection_resist = external_convection['h'] * external_convection['perimeter'] * mesh["d_x"]
    F_e = material_props["rho"] * velocity_field["u"] * mesh["area"]
    F_w = material_props["rho"] * velocity_field["u"] * mesh["area"]

    A, b = np.zeros((n_x, n_x)), np.zeros(n_x)

    for i in range(n_x):
        S_u = vol_heat_source + convection_resist * external_convection['T_conv']
        S_p = -convection_resist
        a_w = thermal_resist + np.max([F_w, 0])
        a_e = thermal_resist + np.max([-F_e, 0])
        if i == 0:
            a_w = 0
            if west_bound['condition'] == 'dirichlet':
                S_p += -(2*thermal_resist + np.max([F_w, 0]))
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

        A[i][i] = a_w+a_e-S_p + (F_e-F_w)
        b[i] = S_u

    return np.linalg.solve(A, b)


