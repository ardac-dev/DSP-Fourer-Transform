import numpy as np

class IIR2ndOrder:

    def __init__ (self, a1, a2, b0, b1, b2):

        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.x = [0, 0, 0]
        self.y = [0, 0, 0]

    
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