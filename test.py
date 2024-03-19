import unittest
import numpy as np
from inertial_forces_v2 import pitch_mat, b2_to_b1

class test_pitch_mat (unittest.TestCase):
    def test_pitch_mat(self):
        vec = np.array([1,1,1])
        pitch_rad = np.deg2rad(90)
        vec_rot = np.transpose(pitch_mat(pitch_rad))@vec
        test = np.allclose(vec_rot, np.array([-1,1,1]),atol=1e-10)
        self.assertTrue(test,"The rotation matrix is wrong.")

    def test_pitch_rot(self):
        vec = np.ones((2*3))
        pitch_rad = np.deg2rad(90)
        vec_rot = b2_to_b1(pitch_rad, vec)
        correct_rot = np.array([-1,1,1,-1,1,1])
        test = np.allclose(vec_rot, correct_rot)
        self.assertTrue(test,"The rotation function is wrong.")

    def test_inertial_loads(self):
        # I don't know how ot test it
        pass

class test_inertial_loads(unittest.TestCase):
    def setUp(self):
        pass



if __name__=="__main__":
    unittest.main()