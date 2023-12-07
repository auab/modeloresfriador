import unittest
import numpy as np
import modelos

class MyTestCase(unittest.TestCase):
    def test_one_dimensional_diffusion_1(self):
        # Consider the problem of source - free heat conduction in an insulated rod
        # whose ends are maintained at constant temperatures of 100°C and 500°C
        # respectively. Calculate the steady state temperature distribution in the rod.
        # Thermal conductivity k equals 1000 W/m.K, cross-sectional area A is 10 × 10−3 m²
        # Solution for 5 cells
        mesh = {"L_x": 0.5, "d_x": 0.1, "area": 0.01}
        west_bound = {'condition': 'dirichlet', 'value': 100}
        east_bound = {'condition': 'dirichlet', 'value': 500}
        material_props = {'k': 1000, 'rho':1}
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion_convection(mesh=mesh, west_bound=west_bound,
                                                                east_bound=east_bound,
                                                                material_props=material_props),
                        np.array([140., 220., 300., 380., 460.])),
                        'The diffusion base case is not correctly solved')

    def test_one_dimensional_diffusion_2(self):
        # A large plate of thickness L = 2cm with constant thermal conductivity
        # k = 0.5 W / m.K and uniform heat generation q = 1000 kW / m3.The faces
        # A and B are at temperatures of 100°C and 200°C respectively.Assuming that
        # the dimensions in the y - and z - directions are so large that temperature
        # gradients are significant in the x-direction only, calculate the steady
        # state temperature distribution.Compare the numerical result with the
        # analytical solution.
        # Solution for 5 cells
        mesh = {"L_x": 0.02, "d_x": 0.004, "area": 1}
        west_bound = {'condition': 'dirichlet','value': 100}
        east_bound = {'condition': 'dirichlet','value': 200}
        material_props = {'k':0.5,'rho':1}
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion_convection(mesh=mesh,
                                                                west_bound=west_bound,
                                                                east_bound=east_bound,
                                                                material_props=material_props,
                                                                q_dot=1e06),
                        np.array([150., 218., 254., 258., 230.])),
                        'The diffusion case with source term is not correctly solved')

    def test_one_dimensional_diffusion_3(self):
        # Convection gives rise to a temperature - dependent heat loss or sink term in the governing
        # equation.Shown in Figure 4.9 is a cylindrical fin with uniform crosssectional area A.The
        # base is at a temperature of 100°C(TA) and the end is insulated.The fin is exposed to an ambient
        # temperature of 20°C. h is the convective heat transfer coefficient, P the perimeter, k the thermal
        # conductivity of the material and T∞ the ambient temperature. Calculate the temperature distribution
        # along the fin and compare the results with the analytical solution
        # Data: L = 1m, hP / (kA) = 25 / m2
        # Solution for 5 cells
        mesh = {"L_x": 1, "d_x": 0.2, "area": 1}
        west_bound = {'condition': 'dirichlet', 'value': 100}
        east_bound = {'condition': 'neumann'}
        material_props = {'k': 1, 'rho':1}
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion_convection(mesh=mesh,
                                                                west_bound=west_bound,
                                                                east_bound=east_bound,
                                                                material_props=material_props,
                                                                perimeter=1, h=25, T_conv=20),
                        np.array([64.22764228, 36.91056911, 26.50406504, 22.60162602, 21.30081301])),
                        'The diffusion case with external convection and neumann condition is not correctly solved')

    def test_one_dimensional_diffusion_convection(self):
        # A property φ is transported by means of convection and diffusion through the
        # one - dimensional domain sketched in Figure 5.2.The governing equation is (5.3);
        # the boundary conditions are φ(0) = 1 at x = 0 and φ(L) = 0 at x = L.Using five equally
        # spaced cells and the central differencing scheme for convectionand diffusion, calculate
        # the distribution of φ as a function of x for u = 0.1m / s,L = 1.0 m, ρ = 1.0 kg/m3,
        # Γ = 0.1 kg/m.s.
        # Solution for 5 cells
        mesh = {"L_x": 1, "d_x": 0.2, "area": 1}
        west_bound = {'condition': 'dirichlet', 'value': 1}
        east_bound = {'condition': 'dirichlet', 'value': 0}
        material_props = {'k': 0.1, 'rho': 1}
        velocity_field = {'u': 0.1}
        self.assertTrue(np.allclose(
            modelos.solve_one_dimensional_diffusion_convection(mesh=mesh,
                                                    west_bound=west_bound,
                                                    east_bound=east_bound,
                                                    material_props=material_props,
                                                    velocity_field=velocity_field),
            np.array([0.93373341, 0.7879469,  0.6130031,  0.40307053, 0.15115145])),
            'The convection diffusion case is not correctly solved')


if __name__ == '__main__':
    unittest.main()
