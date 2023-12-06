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
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion(area=0.01, L=0.5, d_x=0.1,
                                                                west_bound={'condition': 'dirichlet',
                                                                            'value': 100},
                                                                east_bound={'condition': 'dirichlet',
                                                                            'value': 500},
                                                                k=1000),
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
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion(L=0.02, d_x=0.004,
                                                                west_bound={'condition': 'dirichlet',
                                                                            'value': 100},
                                                                east_bound={'condition': 'dirichlet',
                                                                            'value': 200},
                                                                k=0.5, q_dot=1e06),
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
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion(L=1, d_x=0.2,
                                                                west_bound={'condition': 'dirichlet',
                                                                            'value': 100},
                                                                east_bound={'condition': 'neumann'},
                                                                k=1,area=1,perimeter=1, h=25, T_conv=20),
                        np.array([64.22764228, 36.91056911, 26.50406504, 22.60162602, 21.30081301])),
                        'The diffusion case with source term is not correctly solved')
        # Solution for 5 cells

if __name__ == '__main__':
    unittest.main()
