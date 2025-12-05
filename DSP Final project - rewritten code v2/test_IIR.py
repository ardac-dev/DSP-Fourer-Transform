import unittest

from IIR2ndOrder import IIR2ndOrder
from IIRChain import IIRChain


class TestIIR2ndOrder(unittest.TestCase):

    def test_identity_filter(self):
        # b0 = 1, b1 = b2 = 0, a1 = a2 = 0  ->  y[n] = x[n] output should be the same as the input
        
        filt = IIR2ndOrder(a1=0.0, a2=0.0, b0=1.0, b1=0.0, b2=0.0)

        x_seq = [0.0, 1.0, -0.5, 2.0, 0.0]
        for x, expected in zip(x_seq, x_seq):
            y = filt.dofilter(x)
            self.assertAlmostEqual(y, expected, places=7)

    def test_moving_sum(self):
        #b0 = b1 = b2 = 1, a1 = a2 = 0  ->  y[n] = x[n] + x[n-1] + x[n-2]
        #expected output [1, 1, 1, 0, 0, ...]
        filt = IIR2ndOrder(a1=0.0, a2=0.0, b0=1.0, b1=1.0, b2=1.0)

        # impulse input
        x_seq = [1.0, 0.0, 0.0, 0.0, 0.0]
        expected = [1.0, 1.0, 1.0, 0.0, 0.0]

        y_seq = []
        for x in x_seq:
            y_seq.append(filt.dofilter(x))

        for y, e in zip(y_seq, expected):
            self.assertAlmostEqual(y, e, places=7)


class TestIIRChain(unittest.TestCase):

    def test_filter_chain(self):
        
        f1 = IIR2ndOrder(a1=0.0, a2=0.0, b0=1.0, b1=1.0, b2=1.0)
        f2 = IIR2ndOrder(a1=0.0, a2=0.0, b0=1.0, b1=1.0, b2=1.0)

        chain = IIRChain()
        chain.addFilter(f1)
        chain.addFilter(f2)

        # impulse input
        x_seq = [1.0, 0.0, 0.0, 0.0, 0.0]
        # convolution of [1,1,1] with [1,1,1] -> [1,2,3,2,1]
        expected = [1.0, 2.0, 3.0, 2.0, 1.0]

        y_seq = []
        for x in x_seq:
            y_seq.append(chain.dofilter(x))

        for y, e in zip(y_seq, expected):
            self.assertAlmostEqual(y, e, places=7)

if __name__ == "__main__":
    unittest.main()
