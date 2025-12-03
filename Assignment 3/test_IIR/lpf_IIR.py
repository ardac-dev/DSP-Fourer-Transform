import numpy as np

class IIRlpf:


    #2nd-order Butterworth LPF
    # H(s) = 1/(s^2 + sqrt(2)*s + 1)
    def __init__ (self, _cutoff, _fs):

        self.cutoff = _cutoff
        self.fs = _fs
        self.a1 = 0
        self.a2 = 0
        self.b0 = 0
        self.b1 = 0
        self.b2 = 0
        self.x = [0,0,0]
        self.y = [0,0,0]
        self.calc_coeffs()

    
    def calc_coeffs(self):

        alpha = np.sqrt(2)
        
        omega_c = (2*np.pi*self.cutoff)/self.fs

        omega_ac = np.tan(omega_c / 2)

        K = 1 / omega_ac

        A0 = K**2 + alpha*K + 1
        A1 = -2*(K**2) + 2
        A2 = K**2 - alpha*K + 1

        self.b0 = 1/A0
        self.b1 = 2/A0
        self.b2 = 1/A0

        self.a1 = A1/A0
        self.a2 = A2/A0

    def dofilter(self, v):
        
        #shifting values
        self.y[2] = self.y[1]
        self.y[1] = self.y[0]
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]

        #performing output calculation

        self.x[0] = v

        self.y[0] = self.b0 * self.x[0] + self.b1 * self.x[1] + self.b2 * self.x[2] - self.a1 * self.y[1] - self.a2 * self.y[2]

        return self.y[0]





