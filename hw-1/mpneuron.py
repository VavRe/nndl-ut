import numpy as np

class MPNeuron():    
    def __init__(self,bias,weights) -> None:
        self.threshhold = bias    
        self.weights = weights

    def output(self,input):
         return np.dot(input,self.weights)

    def activated_output(self,input):
        output = self.output(input)
        if output >= self.threshhold:
            return 1
        else:
            return 0
