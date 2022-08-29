import numpy as np
import math as mth
import scipy
import scipy.sparse as sp
from scipy import linalg as la
from scipy import special as ss
from scipy.interpolate import interp1d
import itertools as it


# generates a random N x N matrix from one of the Gaussian ensembles
# GOE = real+symm
# GUE = Hermitian
# var = characteristic O(1) variance, σ^2
# output will have diags scaling as σ^2/N and off-diagonals scaling as σ^2/2N
def getWigner(N, var, ensemble='GOE'):
    if ensemble=='GOE' or ensemble=='goe':
        M = np.random.normal(0,np.sqrt(var/(4*N)),size=(N,N)) # var of each entry is σ^2/4N
        # now symmetrize and return output 
        return (M+M.T)/2 # now var of diags will now be 4(σ^2/4N)= σ^2/N and var of off-diags is σ^2/2N
    elif ensemble=='GUE' or ensemble=='gue':
        M  = np.random.normal(0,np.sqrt(var/(4*N)),size=(N,N)) + 1j*np.random.normal(0,np.sqrt(var/(4*N)),size=(N,N))
        # make Hermitian
        return (M+np.conjugate(M.T))/2
    else:
        raise Exception("invalid ensemble!")


# eigs = list of (REAL) eigenvalues
# returns nearest neighbor level spacings
def getLevelSpacings(eigs):
    if not all(eigs.imag < 1e-14):
        raise Exception("Eigenvalues not real!!")
    eigs = np.sort(eigs.real)
    return eigs[1:]-eigs[:-1]


# calculates ratios of neighboring level spacings (which gives level statistics) -- dependence on DoS cancels!
# ∆E_n = E_n - E_{n+1}
# r_n =
def getRatios(eigs):
    eigs = np.sort(eigs)
    r = np.zeros(len(eigs) - 2)

    for i in range(len(eigs) - 2):  # closed interval [0,length(eigs)-3] (since index at 0, this is equiv to [1,length(eigs)-2])
        left = eigs[i] - eigs[i + 1]
        right = eigs[i + 2] - eigs[i + 1]
        r[i] = (eigs[i+2]-eigs[i+1])/(eigs[i+1]-eigs[i])
        if mth.isinf(r[i]):
            print("center idx: %d\t right: %.20f\t left: %.20f\t ratio: %f" % (i, right, left, r[i]))

    #     print(r)
    return r

# uses the minimum of left and right ratios of level spacings
def getMinRatios(eigs):
    eigs = np.sort(eigs)
    right = eigs[2:] - eigs[1:-1]
    left = eigs[1:-1] -  eigs[:-2]
    return np.minimum(left,right)/np.maximum(left,right)


# gives the functional form of analytic level spacing distributions for chosen ensemble

def poissonLS(r):
    return [1 / ((1 + x) ** 2) for x in r]


def wignerLS(r, ensemble='GOE'):
    if ensemble == "GOE":
        b = 1;
        Z = 8 / 27;
    elif ensemble == "GUE":
        b = 2;
        Z = (4 * mth.pi) / (81 * mth.sqrt(3));
    elif ensemble == "GSE":
        b = 4;
        Z = (4 * mth.pi) / (729 * mth.sqrt(3));
    else:
        raise Exception("Invalid ensemble!")

    return [(1 / Z) * ((x + x ** 2) ** b) / ((1 + x + x ** 2) ** (1 + (3 * b / 2))) for x in r]


def logpoissonLS(r):
    return [x / ((1 + x) ** 2) for x in r]


def logwignerLS(r, ensemble='GOE'):
    if ensemble == "GOE":
        b = 1;
        Z = 8 / 27;
    elif ensemble == "GUE":
        b = 2;
        Z = (4 * mth.pi) / (81 * mth.sqrt(3));
    elif ensemble == "GSE":
        b = 4;
        Z = (4 * mth.pi) / (729 * mth.sqrt(3));
    else:
        raise Exception("Invalid ensemble!")

    return [(x / Z) * ((x + x ** 2) ** b) / ((1 + x + x ** 2) ** (1 + (3 * b / 2))) for x in r]



