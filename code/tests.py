import unittest
import numpy as np

from oracles import *
from game import *
# work in progress

class TestLP(unittest.TestCase):
    def test_prisoners_dilema_A(self):
        matrix_PD = np.array([ [ 3, -1],\
                               [-1,  5] ])
        p = optimal_mixed_strategy(matrix_PD)
        self.assertIsNone( np.testing.assert_almost_equal(p, np.array([3/5, 2/5])))

    def test_prisoners_dilema_B(self):
        matrix_PD = np.array([ [ 3, -1],\
                               [-1,  5] ])
        q = optimal_mixed_strategy(matrix_PD)
        self.assertIsNone( np.testing.assert_almost_equal(q, np.array([3/5, 2/5])))

    def test_rock_paper_scissors_A(self):
        matrix_RPS = np.array([ [  0, -1,  1],\
                                [  1,  0, -1],\
                                [ -1,  1,  0] ])
        p = optimal_mixed_strategy(matrix_RPS)
        self.assertIsNone( np.testing.assert_almost_equal(p, np.array([1/3, 1/3, 1/3])))

    def test_rock_paper_scissors_B(self):
        matrix_RPS = np.array([ [  0, -1,  1],\
                                [  1,  0, -1],\
                                [ -1,  1,  0] ])
        q = optimal_mixed_strategy(matrix_RPS, "B")
        self.assertIsNone( np.testing.assert_almost_equal(q, np.array([1/3, 1/3, 1/3])))


class TestBR(unittest.TestCase):
    def test_response_A(self):
        bounds = np.array([[-1, 2]])
        func = lambda x, y: x**2+y**2
        game = Game( HyperBlock(bounds), HyperBlock(bounds), func)

        B = np.array(np.array([0]))
        q = np.array(np.array([1]))

        br, val = bestResponseOracleA(B, q, game)
        self.assertAlmostEqual(br[0], 2, 8, "Should be 0")

    def test_response_B(self):
        bounds = np.array([[-1, 2]])
        func = lambda x, y: (x**2) + (y**2)
        game = Game( HyperBlock(bounds), HyperBlock(bounds), func)

        A = np.array(np.array([-1, 0, 1]))
        p = np.array(np.array([1/3, 1/3, 1/3]))

        br, val = bestResponseOracleB(A, p, game)
        self.assertAlmostEqual(br[0], 0, 8, "Should be 2")


class TestAlreadyExists(unittest.TestCase):
    def test1(self):
        action =   np.array([1, 1])
        actions = np.array([[0, 0],\
                            [0.999, 1]])

        self.assertEqual(already_exists(actions, action, 0.0001), False)
    
    def test2(self):
        action =   np.array([1, 1])
        actions = np.array([[0, 0],\
                            [0.999, 1]])

        self.assertEqual(already_exists(actions, action, 0.01), True)

    def test3(self):
        action =   np.array([1, 1])
        actions = np.array([[0, 0],\
                            [0.999, 1.001]])

        self.assertEqual(already_exists(actions, action, 0.01), True)


class TestReduceStrategies(unittest.TestCase):
    def test1(self):
        xs = np.array( [[0],   [1]] )
        p = np.array([ 1e-9,   1-1e-9 ])

        ys = np.array( [[0],   [1]] )
        q = np.array([ 1-1e-9, 1e-9 ])

        xs_r, p_r, ys_r, q_r = reduce_strategies(xs, p, ys, q, epsilon=1e-8)

        self.assertIsNone( np.testing.assert_almost_equal(xs_r, np.array([[1]])))
        self.assertIsNone( np.testing.assert_almost_equal(p_r, np.array([1])))
        self.assertIsNone( np.testing.assert_almost_equal(ys_r, np.array([[0]])))
        self.assertIsNone( np.testing.assert_almost_equal(q_r, np.array([1])))


if __name__ == '__main__':
    unittest.main()
