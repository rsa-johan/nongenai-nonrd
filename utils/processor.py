import numpy as np
from torch import from_numpy, float32
from polars import DataFrame
from sklearn.preprocessing import StandardScaler

def processor(X, Y):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    Xscaler.fit(X)
    Yscaler.fit(Y)

    def preprocess(x=None, y=None):
        xout, yout = x, y
        if x is not None:
            xout = from_numpy(Xscaler.transform(x)).to(float32)
        if y is not None:
            yout = from_numpy(Yscaler.transform(y)).to(float32)

        return xout, yout

    def postprocess(x=None, y=None):
        xout, yout = x, y
        if x is not None:
            xout = from_numpy(Xscaler.inverse_transform(x)).to(float32)
        if y is not None:
            yout = from_numpy(Yscaler.inverse_transform(y)).to(float32)

        return xout, yout

    return *preprocess(X, Y), preprocess, postprocess
