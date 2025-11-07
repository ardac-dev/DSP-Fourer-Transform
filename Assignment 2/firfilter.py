class FIRfilter:

    def __init__ (self, _coefficients):
        self.coefficients = _coefficients
        self.buffer = [0]*len(_coefficients)
        self.num_taps = len(_coefficients)
        self.offset = 0
    
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








    
