import unittest
import numpy as np
import modelos

class MyTestCase(unittest.TestCase):
    def test_one_dimensional_diffusion_1(self):
        # Consider the problem of source - free heat conduction in an insulated rod
        # whose ends are maintained at constant temperatures of 100°C and 500°C
        # respectively. Calculate the steady state temperature distribution in the rod.
        # Thermal conductivity k equals 1000 W/m.K, cross-sectional area A is 10 × 10−3 m²
        self.assertTrue(np.allclose(
                        modelos.solve_one_dimensional_diffusion(area=0.01, L=0.5, d_x=0.1,
                                                                Ta=100, Tb=500, k=1000),
                        np.array([140., 220., 300., 380., 460.])),
                        'The base case is not correctly solved')



if __name__ == '__main__':
    unittest.main()
