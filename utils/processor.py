import numpy as np
from torch import Tensor
from sklearn.preprocessing import StandardScaler

def processor(X, Y):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    Xscaler.fit(X.to_numpy())
    Yscaler.fit(Y.to_numpy())

    def preprocess(x=None, y=None):
        xout, yout = x, y
        if x.any():
            xout = Tensor(Xscaler.transform(x))
        if y.any():
            yout = Tensor(Yscaler.transform(y))

        return xout, yout

    def postprocess(x=None, y=None):
        xout, yout = x, y
        if x.any():
            xout = Tensor(Xscaler.inverse_transform(x))
        if y.any():
            yout = Tensor(Yscaler.inverse_transform(y))

        return xout, yout

    return *preprocess(X, Y), preprocess, postprocess
