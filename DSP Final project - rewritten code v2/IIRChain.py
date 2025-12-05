import numpy as np
from IIR2ndOrder import IIR2ndOrder
class IIRChain:

    def __init__(self):

        self.filterList = []

    def addFilter(self, f : IIR2ndOrder):

        self.filterList.append(f)
    
    def dofilter(self, x):

        y = x
        for filter in self.filterList:

            y = filter.dofilter(y)
        
        return y