import numpy as np

class FIRfilter:

    def __init__ (self, _coefficients):
        self.coefficients = np.array(_coefficients, dtype=np.float64)
        self.buffer = np.zeros(len(_coefficients), dtype=np.float64)
        self.num_taps = len(_coefficients)
        self.offset = 0
    
    #ring buffer implementation - used for main FIR filter
    def dofilter(self, v):

        output = 0
        
        self.buffer[self.offset] = v

        buf_index = self.offset
        coeff_index = 0

        while(buf_index >= 0 and coeff_index < self.num_taps):
            output += self.buffer[buf_index] * self.coefficients[coeff_index]
            buf_index -= 1
            coeff_index += 1
        
        buf_index = self.num_taps - 1

        while(coeff_index < self.num_taps):
            output += self.buffer[buf_index] * self.coefficients[coeff_index]
            buf_index -= 1
            coeff_index += 1
        
        self.offset += 1

        if (self.offset >= self.num_taps):
            self.offset = 0 
        
        return output
    
    #basic implementation of FIR filter - used for adaptive filtering
    def doFilterSimple(self, v):

        for i in range(self.num_taps - 1, 0, -1):
            self.buffer[i] = self.buffer[i - 1]
        
        
        self.buffer[0] = v

        output = np.inner(self.coefficients, self.buffer)
    
        return output
    
    def doFilterAdaptive(self, signal, noise, learningRate):

        canceller = self.doFilterSimple(noise)

        output_signal = signal - canceller

        for i in range(self.num_taps):
            try:
                self.coefficients[i] = self.coefficients[i] + output_signal*learningRate*(self.buffer[i])
            except:
                print(self.coefficients[i])
        return output_signal
    












    
