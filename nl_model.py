import numpy as np


def linear_(x, params):
    a, b = params
    return a*x+b

def quadratic_(x, params):
    a, b, c = params
    return a*(x**2)+b*x+c


def hill_(x, params):
    Fm, Kd, n, F0 = params
    return Fm*(x**n)/(Kd+x**n)+F0


def sigmoid_(x, params):
    Fm, Ca0, beta, F0 = params
    return Fm/(1+np.exp(-(x-Ca0)*beta))+F0