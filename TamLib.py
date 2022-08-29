from qutip import *
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as mth
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg as sla
from scipy import linalg as la
from scipy import special as ss
from scipy.interpolate import interp1d
import itertools as it
from collections import Counter 
from timeit import default_timer as timer
import time

from matplotlib.collections import PolyCollection # for plotting filled waterfall plots
from matplotlib.collections import LineCollection


# CONVERSIONS ###########################################################################################################################################

# x is a multidimensional array
# construct dict where each of the axes yields a column for the idx, plus a last column for the data
# clabels is a DICT that contains the names of the axes, e.g. {0: 'i', 1:'j', 2='k'}
# Note: the name of the data col is referred to by axis idx len(x.shape), e.g. x = 2x3x5 ==> data col idxed by 3
# axesrange is a DICT that contains the ORDERED values that each idx value maps to, e.g. can have axis0 idcs 1,2,3 --> axis0 vals -1,0,-1
def arr2df(x, clabels = {}, axesrange={}):    
    for n in set(range(len(x.shape)+1)).difference(set(clabels.keys())):
        clabels.update({n: chr(97+n)})
    
    for n in set(range(len(x.shape))).difference(set(axesrange.keys())):
        axesrange.update({n: range(x.shape[n])})
    
    # convert dictionaries into an ordered list of axes values and col labels
    clabels = [i[1] for i in sorted(clabels.items())]
    axesvals = [i[1] for i in sorted(axesrange.items())]
    
    cols = list(zip(*it.product(*axesvals)))
    cols.append(tuple(x.ravel()))
    
    dataDict = dict(zip(clabels,cols))
    df = pd.DataFrame.from_dict(dataDict)
    
    return df

# Input: df of n cols where first n-1 cols are index cols and last col is data col
# Convert to multidimensional array
def df2arr(df):
    header = list(df.columns)
    grouped = df.groupby(header[:-1])[header[-1]].mean()
    arr = np.full(tuple(map(len,grouped.index.levels)),np.nan) # create empty NaN array of appropriate size
    arr[tuple(grouped.index.codes)] = grouped.values.flat
    return arr

def maxposdiff(a,b):
    return np.max(np.abs(a-b))

# returns INTEGER-VALUED ceil(a/b)
def ceildiv(a,b):
    return -(a//-b)

# turns fractions into a string (useful for filenames). e.g. 0.5 --> 0p5
def dec2str(x):
    return str(x).replace('.','p')

def str2dec(s):
    arr = np.array(s.split('p')).astype('int')
    if len(arr)==1:
        return arr[0]
    else:
        return arr[0]+arr[1]/(10**np.ceil(np.log10(arr[1])))

# GENERAL UTILITY ###########################################################################################################################################

# converts a number x into its binary representation with N bits
def dec2bin(x,N):
    return ('{:0%db}'%N).format(x)

# N = num bits
# x = integer representation of a binary configuration of N bits
# direction = direction which to shift bits
def shiftBits(x,N,direction='left'):
    x = x%(2**N-1)  # ensures that x is a valid integer representation
    if direction=='left':
        return ((n<<d) & (2**N-1)) | ((n & (2**N-1)) >> (N-d) )
    if direction=='right':
        return ((n & (2**N - 1)) >> d) | ((n << (N-d)) & (2**N-1))

def gaussianFunc(x, mean, A, var):
    return A*np.exp(- (x - mean)**2 / (2*var))

# CARTESIAN: params = (mean real, var real, mean imag, var imag)
# POLAR: params = (mean R, std R), angle is chosen uniformly
def getRandomGaussComplex(params,shape):
    if len(params)==2:  # POLAR
        return np.random.normal(loc=params[0],scale=np.sqrt(params[1]),size=shape)*np.exp(1j*np.random.uniform(low=-np.pi,high=np.pi,size=shape))
    else:   # CARTESIAN
        return np.random.normal(loc=params[0],scale=np.sqrt(params[1]),size=shape) + 1j*np.random.normal(loc=params[2],scale=np.sqrt(params[3]),size=shape)

# CARTESIAN: params = (lowBnd_real, highBnd_real, lowBnd_imag, highBnd_imag) OR (lowBnd R, highBnd R)
# POLAR: params = (mean R, std R), angle is chosen uniformly
def getUniform(params,shape):
    if len(params)==2: # POLAR
        return np.random.uniform(low=params[0],high=params[1],size=shape)*np.exp(1j*np.random.uniform(low=-np.pi,high=np.pi,size=shape))
    else:
        return np.random.uniform(low=params[0],high=params[1],size=shape)+ 1j*np.random.uniform(low=params[2],high=params[3],size=shape)

    
# get all possible binary vectors of size n
def getBinCombos(n,returnArray=False):
    if returnArray: 
        return np.array(list(it.product([0, 1], repeat=n)))
    else:
        return list(it.product([0, 1], repeat=n))

# same as above but returns a vec composed of {-1,1}
def getUpDownCombos(n,returnArray=False):
    if returnArray: 
        return np.array(list(it.product([-1, 1], repeat=n)))
    else:
        return list(it.product([-1, 1], repeat=n))


# a = number; n = number of bits in representtion
def flipBits(a,n,returnType="int"):
    if returnType=="int":
        return int(bin((a ^ (2 **(n+1) - 1)))[3:],2)
    else:
        return bin((a ^ (2 **(n+1) - 1)))[3:]

# express the state n = # as a list of 2**L bits
def bitsToList(n,L):
    return list(map(int,list(np.binary_repr(n,2**L))))

# takes in two matrices and computes direct sum
def directSum(A, B):
    dsum = np.zeros(np.add(A.shape, B.shape), dtype=complex)
    dsum[:A.shape[0], :A.shape[1]] = A
    dsum[A.shape[0]:, B.shape[1]:] = B
    return dsum

# circularly shifts lists by n elements
def circshift(arr,n=1,direction="left"):
    if direction=="left":
        return arr[n::] + arr[:n:] 
    else:
        return arr[-n:] + arr[0:-n] 

# checks if two lists have same cyclic order
def checkCyclic(l1, l2):
    if len(l1) != len(l2):
        raise Exception("Lists have different size!")
    if not (set(l1)==set(l2)):
        raise Exception("Lists have different elements!")
    
    first = l1[0]
    idx = l2.index(first)
    cycleL2 = circshift(l2,idx)
    return l1==cycleL2

# given two lists with same elements return levi civita
def levicivita(l):
    cyclic = np.arange(min(l),max(3,max(l))+1,step=1).tolist()
    setDiff = np.setdiff1d(cyclic,l)
    lorder = l.copy()
    lorder.extend(setDiff)
    returnVals = [-1,1]   # TRUE = 1, FALSE = 0
    return returnVals[checkCyclic(lorder,cyclic)]


# finds the relative angle between two angular positions
def getThetaRel(th1,th2):
    return abs((th1%(2*np.pi))-th2%(2*np.pi))

# get list of coordinates as a dxN array where N is the total number of lattice points
# d = dimension
# a = lattice spacing along each axis/dimension; if is a number, then uniform spacing
# L = number of sites along each axis/dimension; if is a number, then square grid; ow L is dx1 vector giving dimensions
def getLatticeCoord(d,L,a):
    L = L*np.ones(d)
    a = a*np.ones(d)
    coordvecs = tuple([np.arange(L[i]*a[i],step=a[i]) for i in range(d)])
    coord = np.meshgrid(*coordvecs)
    return np.stack(list(map(np.ravel, coord)),axis=1)


# coord = Lx3 matrix where L = # of spins
# returns the distance and cos(th) of all pairs of ising spins from coordinates
def get3DThetaDistCoord(coord):
    xcoord = coord[:,0]
    ycoord = coord[:,1]
    zcoord = coord[:,2]
    
    # displacement coordinates
    xi,xj = np.meshgrid(xcoord,xcoord,sparse=True)
    yi,yj = np.meshgrid(ycoord,ycoord,sparse=True)
    zi,zj = np.meshgrid(zcoord,zcoord,sparse=True)
    zd = zi - zj
    r_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zd)**2)  # mag of displacement
    costh_ij = np.divide(zi-zj,r_ij + np.eye(r_ij.shape[0]),out=np.zeros_like(r_ij),where=(zi-zj)!=0)
    return r_ij,costh_ij

# returns matrix of distances between each pair of spins
def pairwiseDist(coord):
    sqdist = 0
    for i in range(coord.shape[1]):
        xi,xj = np.meshgrid(coord[:,i],coord[:,i],sparse=True)
        sqdist = sqdist + (xi-xj)**2
    r_ij = np.sqrt(sqdist)
    return r_ij

# PLOTTING ###########################################################################################################################################

# returns image of matrix with nonzero elements colored black
def getIm(H):
    Im = np.logical_not(abs(H - 0) < 1e-13);
    return Im

# Construct the vertex list which defines the polygon filling the space under the (x, y) line graph
# This assumes x is in ascending order.
def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


# FILE I/O #################################################################################################

# reads multidimensional array from txt file into data frame
# returns df and header = list of col names
def load2df(filename,footer=False):
    with open(filename, 'r') as f:
        # Load into dataframe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if footer: # Get header separately if have a footer marked with # 
            with open(filename) as f:
                headerline = f.readline()
            header = headerline.strip().lstrip("#").split()
            df = pd.read_csv(filename,delimiter='\t',names=header,comment='#',index_col=False)
        else:
            df = pd.read_csv(filename,delimiter='\t',escapechar="#",index_col=False)
            df.columns = df.columns.str.strip()
            header = list(df.columns)
        df[header[-1]] = df[header[-1]].apply(lambda x: round(x,16)) # round last col = data col to 16 decimal places, the maximum precision given when outputting to txt file as str

    return df, header

# reads multidimensional array data from txt file into array
# assumes last col = array entries and the 1st through 2nd to last col are indices
def load2arr(f):
    data = np.loadtxt(f)
    return df2arr(pd.DataFrame(data)) # convert to df so can organize indices, then convert back into arr


# given some 3D matrix Z with rows indexed by values r and cols by values c
def export3D(r,c,Z,filestr, header="",footer=""):
    R,C = zip(*product(r,c))
    df = pd.DataFrame().assign(x=R,y=C,z=Z.ravel())
    np.savetxt(filestr, df.values, fmt='%f',header=header,footer=footer)

# exports multidimensional array to text file labeled by indices
# idcs = (optional) dictionary of index ranges; key = axis #; value = range of values (e.g. for a p x q array, idcs = {0: range(p), 1: range(q)}
# if want data to be labeled by values other than idx coordinates, can let idcs = [values assoc to axis 0 idcs, values assoc to axis 1 idcs, ...]
def exportNdArray(M, filestr, idcs=[], clabels=[],header="",footer="",delim='\t'):
    if not clabels:
        clabels = [chr(x) for x in np.arange(97,97+len(M.shape))] + ['data']
    if not header:
        header = (delim).join(clabels)
    idx_ranges = []
    for axis in range(len(M.shape)):
        idx_ranges.append(idcs[axis]) if axis in idcs else idx_ranges.append(range(M.shape[axis]))

    I = list(zip(*it.product(*idx_ranges))) # each element in the list is a column vector containing the column data
    I.append(tuple(M.ravel()))
    dataDict = dict(zip(clabels,I))
    df = pd.DataFrame.from_dict(dataDict)
    np.savetxt(filestr, df.values, fmt=['%d']*len(idx_ranges)+['%s'], delimiter=delim,header=header,footer=footer)


# assumes for each key have a value which can generically be a multidimensional array A with n axes
# indexing the values of the data by (i_1,i_2,...i_n, k) where k is the key value
# cols are "i_1, i_2, .... i_n, k, dict[k][0,1,...n]"
def exportDict(dict, filestr, clabels=[],header="",footer="",delim='\t'):
    arrShape = list(dict.values())[0].shape
    if not clabels:
        clabels = [chr(x) for x in np.arange(97,97+len(arrShape)+1)] + ['data']
    if not header:
        header = (delim).join(clabels)

    M = np.array(list(dict.values())) # np array where 0th axis is the keys
    M = M.transpose(list(range(1,len(M.shape)))+[0]) # make keys LAST axis

    idcs = [range(x) for x in M.shape[:-1]]
    idcs = idcs + [list(dict.keys())]

    exportNdArray(M, filestr, idcs=idcs, clabels=clabels,header=header,footer=footer,delim=delim)


# OPERATOR MECHANICS HELPERS ###########################################################################################################################################

# for numpy operators
def isDiag(M):
    i, j = M.shape
    assert i == j 
    test = M.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

# Hermitian conjugate of 2d numpy array
def Hconj(op):
    return np.conjugate(np.transpose(op))

# outer product of 1d numpy arrays (vectors)
def outerprod(v1,v2):
    return np.outer(v1,np.conjugate(v2))

# returns a PRODUCT of many-body operators, each elem in op_list acts on tensor of L sites
# opList = LIST of np arrays, Qobjs, or sparse matrices
def opProd(opList):
    if isinstance(opList[0],Qobj):
        P = qeye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P*opList[i]
        return P
    elif isinstance(opList[0],sp.csr_matrix):
        P = sp.eye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P*opList[i]
        return P
    elif isinstance(opList[0],np.ndarray):
        P = np.eye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P@opList[i]
        return P
        # return np.linalg.multi_dot(opList)
    else:
        raise Exception("unknown type")

opProdFunc = np.vectorize(opProd)


# define many-body operator on N qubits from list of LOCAL qubit ops at locations given by pos
# pos indexed at 0
def tensorOp(N, op, pos):
    idxCheck = len(op) == len(pos)  # make sure op and pos have same size
    if not (idxCheck):
        raise Exception('Number of operators doesn''t match number of positions')
    opList = [qeye(2)]*(pos[0]) + [op[0]]
    
    for i in range(1,len(op)):
        opList = opList +[qeye(2)]*(pos[i]-pos[i-1]-1) + [op[i]]
    opList = opList + [qeye(2)]*(N-pos[-1]-1)
    return tensor(opList)


# define many-body operator on N qubits from list of LOCAL qubit ops at locations given by pos
# operators are Qobjs
# pos indexed at 0
def tensorList(N, op, pos):
    idxCheck = len(op) == len(pos)  # make sure op and pos have same size
    if not (idxCheck):
        raise Exception('Number of operators doesn''t match number of positions')

    if not (max(pos)<=N):
        raise Exception('One or more position arguments exceed max number of qubits')

    order = np.argsort(pos)
    pos = pos[order]
    op = pos[order]

    opList = [qeye(2)]*(pos[0]) + [op[0]]
    
    for i in range(1,len(op)):
        opList = opList +[qeye(2)]*(pos[i]-pos[i-1]-1) + [op[i]]
    opList = opList + [qeye(2)]*(N-pos[-1]-1)
    return tensor(opList)


# Hilbert Schmidt inner prod for Majoranas and ops defined using Maj basis
# normalized using {χ_i, χ_j} = δ_ij ==> χ_i^2 = 1/2
# op1, op2 = BOTH qobjs, sparse csr, or np arrays
def schmidt_Maj(op1,op2):
    if type(op1)!=type(op2):
        raise Exception("Operators not same type! Have %s and %s" % (type(op1),type(op2)))

    if isinstance(op1,Qobj) and isinstance(op2,Qobj):
        inp = (2/(op1.shape[0]))*(op1.dag()*op2).tr()

    elif isinstance(op1,sp.csr_matrix) and isinstance(op2,sp.csr_matrix):
        inp = (2/(op1.shape[0]))*(op1.H*op2).diagonal().sum()

    else: # np array
        inp = ( (2/(op1.shape[0]))*((op1.conjugate().transpose())@op2)).trace()

    return round(inp.real,15) + 1j*round(inp.imag,15)


# EXPECTATION VALUES AND CORREL FUNCS ###########################################################################################################################################

# op = qobj, np array, or sparse
# eigs = np array
# evecs = 2d np array
# returns sparse matrix
def timeEvolve(op,t,eigs,evecs,energyBasis = False):
    Ut_diag =  sp.diags(np.exp(-1j*eigs*t))
    Ut_diag_conj = sp.diags(np.exp(1j*eigs*t))
    evecs_sp = sp.csr_matrix(evecs)
    if isinstance(op,Qobj):
        op_e = evecs_sp.H * op.data * evecs_sp
    elif isinstance(op,np.ndarray):
        op_e = evecs_sp.H * sp.csr_matrix(op) * evecs_sp
    else: # sparse
        op_e = evecs_sp.H * op * evecs_sp

    if energyBasis:
        return Ut_diag_conj * op_e * Ut_diag
    else:
        return evecs_sp * Ut_diag_conj * op_e * Ut_diag * evecs_sp.H

def timeEvolve_sp(op,t,eigs,evecs, energyBasis = False):
    Ut_diag =  sp.diags(np.exp(-1j*eigs*t))
    Ut_diag_conj = sp.diags(np.exp(1j*eigs*t))
    evecs_sp = sp.csr_matrix(evecs)
    op_e = evecs_sp.H * op * evecs_sp
    if energyBasis:
        return Ut_diag_conj * op_e * Ut_diag
    else:
        return evecs_sp * Ut_diag_conj * op_e * Ut_diag * evecs_sp.H


def timeEvolve_np(op,t,eigs,evecs,energyBasis = False):
    Ut_diag =  np.diag(np.exp(-1j*eigs*t))
    Ut_diag_conj = np.diag(np.exp(1j*eigs*t))

    op_e = evecs.conjugate().T @ op @ evecs
    
    if energyBasis:
        return Ut_diag_conj @ op_e @ Ut_diag
    else:
        return evecs @ Ut_diag_conj @ op_e @ Ut_diag @ evecs.conjugate().T


# FASTEST
# Gets thermal2pt func b/t time evolved op1 and op2 at t=0, (1/Z) Tr(e^(-βH) O1(t) O2 )
# input ops are np arrays
# input ops are np arrays
def thermal2pt(op1,op2,t,beta,eigs,evecs):
    op1_e = evecs.conjugate().T @ op1 @ evecs
    op2_e = evecs.conjugate().T @ op2 @ evecs
    Z = np.sum(np.exp(-beta*eigs))
    return (1/Z)*np.einsum('ii,jj,ij,ji',np.diag(np.exp(-(beta-1j*t)*eigs)), np.diag(np.exp(-1j*eigs*t)),op1_e, op2_e)


# returns thermal 2 point function Tr(e^(-βH) O_1 O_2 )
# tvec = number
# op1 and op2 are BOTH sparse, np arrays, or Qobj
# eigs, evecs from diagonalizing the Hamiltonian
# eigs = 1d np array; evecs = 2d np array with each cols an eigenvectors
def thermal2pt_general(op1,op2,beta,eigs,evecs):
    if type(op1)!=type(op2) and (sp.issparse(op1)!=sp.issparse(op2)):
        raise Exception("Operators have diff types!")

    Z = np.sum(np.exp(-beta*eigs))

    if isinstance(op1,sp.csr_matrix) or isinstance(op1,sp.csc_matrix) or isinstance(op1,qutip.fastsparse.fast_csr_matrix):
        evecs_sp = sp.csr_matrix(evecs)
        op12_e = evecs_sp.H * op1 * op2 * evecs_sp
        return (1/Z)*( (sp.diags(np.exp(-beta*eigs)) * op12_e ).diagonal().sum() )

    elif isinstance(op1,np.ndarray):
        op12_e = evecs.conjugate().transpose() @ op1 @ op2 @ evecs
        return (1/Z)*(( np.diag(np.exp(-beta*eigs)) @ op12_e ).trace())

    elif isinstance(op1,Qobj):
        evecs_q = Qobj(evecs)
        op12_e = evecs_q.dag() * op1 * op2 * evecs_q
        return (1/Z)*( (Qobj(sp.diags(np.exp(-beta*eigs))) * op12_e ).tr() )


def thermal2pt_sp(op1,op2,beta,eigs,evecs):
    return  ((sp.diags(np.exp(-beta*eigs)) *sp.csr_matrix(evecs).H * (op1*op2) * sp.csr_matrix(evecs)).diagonal().sum())/np.sum(np.exp(-beta*eigs)) 

def thermal2pt_np(op1,op2,beta,eigs,evecs):
    return  ((np.diag(np.exp(-beta*eigs))@ evecs.conjugate().transpose()@ (op1@op2)@ evecs).trace())/np.sum(np.exp(-beta*eigs)) 

thermal2ptFunc = np.vectorize(thermal2pt_np,signature='(m,n),(m,n),(),(k),(q,r)->()')

# returns thermal 2 point function bt 2 ops from list Tr(e^(-βH) O_j(t) O_k )
# opList is a list of np arrays
# j,k are the indices of the ops in the list
def thermal2ptList(j,k,t,opList,beta,eigs,evecs):
    return ((np.diag(np.exp(-beta*eigs))@ evecs.conjugate().transpose()@ (timeEvolve(opList[j],t,eigs,evecs).A@opList[k])@ evecs).trace())/np.sum(np.exp(-beta*eigs))

thermal2ptFuncList = np.vectorize(thermal2ptList,excluded=[3,4,5,6])


# returns thermal 2 point function Tr(e^(-βH) O_1 O_2 )
# tvec = number
# op1 and op2 are BOTH SPARSE
# eigs, evecs from diagonalizing the Hamiltonian
# eigs = 1d np array; evecs = 2d np array with each cols an eigenvectors
def thermal2pt_q(op1,op2,beta,eigs,evecs):
    evecs_sp = sp.csr_matrix(evecs)
    op12_e = evecs_sp.H * op1 * op2 * evecs_sp
    Z = np.sum(np.exp(-beta*eigs))
    return (1/Z)*( (sp.diags(np.exp(-beta*eigs)) * op12_e ).diagonal().sum() )


def thermal2ptFT(op1,op2,beta,eigs,evecs):
    op1_e = np.conjugate(evecs) @ op1 @ np.transpose(evecs)
    op2_e = np.conjugate(evecs) @ op2 @ np.transpose(evecs)
#     print(Qobj(op1_e))
    gdict = {}
    for m,n in it.combinations_with_replacement(range(len(eigs)),2):
        w = eigs[n]-eigs[m]
        if w in gdict.keys():
            gdict[w] = gdict[w]+(np.exp(-beta*eigs[n]) + np.exp(-beta*eigs[m]))*2*np.pi*(op1_e[n,m]*op2_e[m,n] + op1_e[n,m]*op2_e[m,n])
        else:
            gdict[w] = (np.exp(-beta*eigs[n]) + np.exp(-beta*eigs[m]))*2*np.pi*(op1_e[n,m]*op2_e[m,n] + op1_e[n,m]*op2_e[m,n])
    wvec_pos = np.round(np.array(list(gdict.keys())),8)
    twoPtFT_pos = np.round(np.array(list(gdict.values())),8)
    sortOrder = np.argsort(wvec_pos)
    wvec_pos = wvec_pos[sortOrder]
    twoPtFT_pos = twoPtFT_pos[sortOrder]
    if not np.allclose(twoPtFT_pos,twoPtFT_pos.real):
        raise Exception("Imaginary G twiddle!")
    return wvec_pos, twoPtFT_pos.real


# Computes G_jk(t) = < χ_j(t), χ_k(0) >_β where the χ_j's are operators
# MEASURES TIME IN UNITS OF ∆t (tdel)!
# ops = N x [opdim] numpy array of operators at t=0
# eigs,evecs = derived from Hamiltonian
# Returns G_jk(t) = N x N matrix of thermal 2pt functions at a single t
def getGjk(t,ops,beta,eigs,evecs,tdel=1):
    N = ops.shape[0]
    Gjk_vec = np.array([thermal2pt(jOp,kOp,t*tdel,beta,eigs,evecs) for  jOp,kOp in it.product(ops, ops)], dtype='complex128')
    Gjk = np.reshape(Gjk_vec,(N,N))
    return Gjk


# GENERAL SYMMETRY HELPERS ###########################################################################################################################################

# symmetry block
class block:
    def __init__(self, p=None):
        self.val = p
        self.states = []
        self.blockSize = None

# given a symmetry operator, find its eigenvalues (one for each block) and eigenvectors
# symm.op and symm.COB are qobj
class symmetry:
    def __init__(self, Op):
        self.op = Op  # actual symmetry operator
        self.COB = []   # matrix of eigenvecs that makes symm op diagonal
#         self.membership = {}  # a dictionary that says which sector each of the eigenstates belongs to; key = idx of eigenstate, value = symm eigenval
        self.blocks = {}    # dictionary containing eigenvalue of each symmetry sector and number of states
        self.blockSizes = None
        self.minBlockSize = None  # size of smallest block
        return
    
    # diagonalize symmetry operator if can't do it by construction
    def fillSymm(self):
        (eigs,states) = self.op.eigenstates()
        eigs = np.round(eigs,10) # round eigenvalues
        self.membership = dict(zip(range(len(states)), eigs)) 
        A = [np.array(x) for x in states]
        self.COB = np.asmatrix(np.array(A))
        self.blocks = dict(Counter(eigs))
        self.blockSizes = self.blocks.values()
        self.minBlockSize = min(self.blocks.values())
        

# for each symmetry of Hamiltonian, have COB. Then block diagonalize H using COB and calculate eigenstates in this basis.
# for each eigenstate in symm basis have symm sector valuend

# class to store a Hamiltonian and its associated symmetries and block diagonal form
class Hsymm_class:
    def __init__(self, H, symm):
        self.H = Qobj(H)  # qobj
        self.HBD = []
        self.BDSymm = []
        self.symmetries = []
        self.addSymm(symm)  # symm = instantiation of class
        self.eigs = []
        self.evecs = []
        self.beigs = []
        self.bvecs = []

    def addSymm(self, symm):
        comm = commutator(self.H,symm.op)
        if comm!=Qobj(np.zeros(self.H.shape)):
            raise Exception("not a symmetry!")
        self.symmetries.append(symm)
        # sort symmetries based on which one has highest "resolution" of state, aka one with the smallest minimal block
        self.symmetries.sort(key=lambda x: x.minBlockSize)

        # update block diagonal matrix
        # choose the symmetry with the "finest granularity"
        self.BDSymm = self.symmetries[0]
        S = self.BDSymm.COB.data  # qobj --> qsparse
        # self.HBD = S @ np.array(self.H) @ S.H
        self.HBD = S.getH()*self.H.data*S  # HBD = qsparse matrix; getH() = Hermitian conjugate


    # diagonalize block diagonal Hamiltonian
    def diagonalize(self):
        blockSizes = self.BDSymm.blockSizes  # dictionary that contains the symmetry blocks
        rank = self.H.shape[0]
        beigs = []  # list of lists that will contain the eigenvalues of each block
        bvecs = [] # list of lists that will contain the eigenvectors of each block
        Hdim = max(self.HBD.shape)

        end = 0
        for bs in blockSizes:
            start = end
            end = end + bs
            hBlock = self.HBD[start:end, start:end]

            # diagonalize the H block
            eigs,evecs = np.linalg.eigh(hBlock.toarray())

            # pad with 0's to get full eigenvectors
            before = start
            after = Hdim - end
            evecs_full = [np.pad(v,(before,after))  for v in np.transpose(evecs)]  # each _ROW_ of evecs_full is an eigenvector of HBD

            beigs.append(eigs)   
            bvecs.append(evecs_full)

        beigs_tot = np.concatenate(beigs) 
        bvecs_tot = np.concatenate(np.array(bvecs))  # _ROWS_ are eigenvectors of _HBD_
        bvecs_tot = np.round(self.BDSymm.COB*Qobj(bvecs_tot.T),15)  # convert to eigenvectors of H + take transpose so COLS are evecs

        order = np.argsort(beigs_tot)
        beigs_tot = beigs_tot[order]
        bvecs_tot = bvecs_tot[:,order]
        
        self.beigs = beigs
        self.bvecs = bvecs
        self.eigs = beigs_tot
        self.evecs = bvecs_tot

# diagonalize block diagonal Hamiltonian given a dictionary of symmetry blocks with their values and sizes
# EIGENVECTORS ARE **ROWS**
def diagonalizeHBD(HBD,blockSizes,COB):
#     totEigs = []  # list of all the eigenvalues of the block diagonal Hamiltonian
    beigs = []  # list of lists that will contain the eigenvalues of each block
    bvecs = []
    Hdim = max(HBD.shape)
    end = 0
    for bs in blockSizes:
        start = end
        end = end + bs
        currBlock = np.array(HBD[start:end, start:end])
#         print(type(currBlock))

        # diagonalize the block
        eigs, evecs = np.linalg.eigh(currBlock)
        beigs.append(eigs)
        
        # pad with 0's to get full eigenvectors
        before = start
        after = Hdim - end
        
#         evecs_full = [np.array(COB.H)@(np.pad(v,(before,after)))  for v in np.transpose(evecs)]
        evecs_full = [np.round(np.pad(v,(before,after)),15)  for v in np.transpose(evecs)]

        bvecs.append(evecs_full)

    beigs_tot = np.concatenate(beigs) 
    bvecs_tot = np.concatenate(np.array(bvecs))
#     bvecs_tot = np.transpose(np.array(COB.H)@np.transpose(bvecs_tot))
    bvecs_tot = bvecs_tot@np.array(COB.conjugate())
    order = np.argsort(beigs_tot)
    beigs_tot = beigs_tot[order]
    bvecs_tot = bvecs_tot[order] # sort rows = eigenvectors

    return (beigs_tot, bvecs_tot, beigs, bvecs)


# LEVEL STATS HELPERS #################################################################################################

def getLevelSpacings(eigvals):
    CHECK_EIG = all(eigvals.imag - 0 < 1e-14)
    if not CHECK_EIG:
        raise Exception("Eigenvalues not real!!")
    eigvals = np.sort(eigvals.real)
    diff = [eigvals[i + 1] - eigvals[i] for i in range(0, len(eigvals) - 1)]
    return diff


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

# uses the minimum of left and right ratios of level spacings
def getMinRatiosLS(s):
    minRatio = [min(s[i],s[i+1])/max(s[i],s[i+1]) for i in range(len(s)-1)]
    return minRatio

def getLSandRatios(eigs):
    s = getEigDiff(eigs)
    r = getMinRatiosLS(s)
    return s,r 

# get NxN Gaussian orthogonal matrix, diagonalize to obtain eigenvalues, and calc level spacings
# do this niter times to build up statistics
# consider only a fraction f of the eigenvalues
def getGOE(N,niter=100,f = 1,diagnostic="spacings"):
    delta = []
    for n in range(niter):
        a = np.random.normal(size=(N,N))
        M = np.tril(a) + np.tril(a,-1).T
        evals = linalg.eigvalsh(M)
        
        start = int(np.rint(((1-f)/2)*N))
        end = int(np.rint(((1+f)/2)*N))
        evals_trunc = evals[start:end]   # consider only middle fraction of eigenvalues
        
        if diagnostic=="spacings":
            delta.extend(getEigDiff(evals_trunc))
        elif diagnostic=="ratios":
            delta.extend(getRatios(evals_trunc))
        elif diagnostic=="minratios":
            delta.extend(getMinRatios(evals_trunc))
        else:
            raise Exception("Invalid diagonstic")
    return delta

def getGOERatios(N,niter=100,f = 1):
    r = []
    minR = []
    for n in range(niter):
        a = np.random.normal(size=(N,N))
        M = np.tril(a) + np.tril(a,-1).T
        evals = linalg.eigvalsh(M)
        
        start = int(np.rint(((1-f)/2)*N))
        end = int(np.rint(((1+f)/2)*N))
        evals_trunc = evals[start:end]   # consider only middle fraction of eigenvalues
        
        r.extend(getRatios(evals_trunc))
        minR.extend(getMinRatios(evals_trunc))
    return r,minR
