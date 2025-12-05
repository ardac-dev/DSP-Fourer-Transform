import numpy as np

def butter_lpf_2nd(cutoff, fs):

        alpha = np.sqrt(2)
        
        omega_c = (2*np.pi*cutoff)/fs

        omega_ac = np.tan(omega_c / 2)

        K = 1 / omega_ac

        A0 = K**2 + alpha*K + 1
        A1 = -2*(K**2) + 2
        A2 = K**2 - alpha*K + 1

        b0 = 1/A0
        b1 = 2/A0
        b2 = 1/A0

        a1 = A1/A0
        a2 = A2/A0

        return a1,a2,b0,b1,b2

def butter_hpf_2nd(cutoff, fs):
        
        alpha = np.sqrt(2)
        
        omega_c = (2*np.pi*cutoff)/fs      # digital cutoff
        omega_ac = np.tan(omega_c / 2)               # prewarped analog cutoff

        K = 1 / omega_ac

        A0 = K**2 + alpha*K + 1
        A1 = -2*(K**2) + 2
        A2 = K**2 - alpha*K + 1

        # High-pass numerator
        b0 = (K**2) / A0
        b1 = -2*(K**2) / A0
        b2 = (K**2) / A0

        # Denominator (same poles as LPF)
        a1 = A1 / A0
        a2 = A2 / A0

        return a1, a2, b0, b1, b2

