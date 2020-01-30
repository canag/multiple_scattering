import pytest 
import numpy as np
from numpy.linalg import norm
import collective_Smatrix as tools
import platonic_solid as positions

# test on CG coefficients
# -----------------------

# literal expressions taken from wikipedia at
# http://en.wikipedia.org/wiki/Table_of_Clebsch–Gordan_coefficients

def test_CGcoeff1010():
    a = np.array([-1/np.sqrt(3), 0, np.sqrt(2/3), 0])
    b = tools.CG_coeff(1, 0, 1, 0)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff111m1():
    a = np.array([np.sqrt(1/3), np.sqrt(1/2), np.sqrt(1/6), 0])
    b = tools.CG_coeff(1, 1, 1, -1)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff1m111():
    a = np.array([np.sqrt(1/3), -np.sqrt(1/2), np.sqrt(1/6), 0])
    b = tools.CG_coeff(1, -1, 1, 1)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff2010():
    a = np.array([0, -np.sqrt(2/5), 0, np.sqrt(3/5), 0])
    b = tools.CG_coeff(2, 0, 1, 0)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff2111():
    a = np.array([0, 0, -np.sqrt(1/3), np.sqrt(2/3), 0])
    b = tools.CG_coeff(2, 1, 1, 1)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff2210():
    a = np.array([0, 0, np.sqrt(2/3), np.sqrt(1/3), 0])
    b = tools.CG_coeff(2, 2, 1, 0)
    assert (norm(a-b)/norm(a))<1e-10

def test_CGcoeff2020():
    a = np.array([np.sqrt(1/5), 0, -np.sqrt(2/7), 0, np.sqrt(18/35), 0])
    b = tools.CG_coeff(2, 0, 2, 0)
    assert (norm(a-b)/norm(a))<1e-10


# test on Platonic solids
# -----------------------

def test_platonic_dist_to_center():
    cond = True
    for N in [5, 7, 9, 13, 21]:
        ka = np.random.uniform(1,5)
        pos = np.zeros((3,N))
        pos[:, :-1] = positions.platonic(N-1, ka)
        for i in range(N-1):
            cond = cond & ((norm(pos[:,i]-pos[:,-1])/ka-1)<1e-10)
    
    assert cond

# test the length of edges

def test_platonic_N4():
    N = 4
    nei = 3 # number of neighbors for each vertex
    val = np.sqrt(8/3) # expected value for unity solid

    ka = np.random.uniform(1,5)
    valka = val*ka
    pos = positions.platonic(N, ka)

    vertices_dist = []
    for i in range(N):
        for j in range(N):
            if i<j:
                vertices_dist.append(norm(pos[:,i]-pos[:,j]))
    vertices_dist.sort()

    n = int(N*nei/2)
    cond_1 = (np.array([norm(v-valka) for v in vertices_dist[:n]])<1e-10).all()
    if N>n:
        cond_2 = (vertices_dist[n]-valka)>1e-10
    else:
        cond_2 = True

    assert cond_1 & cond_2


def test_platonic_N6():
    N = 6
    nei = 4 # number of neighbors for each vertex
    val = np.sqrt(2) # expected value for unity solid

    ka = np.random.uniform(1,5)
    valka = val*ka
    pos = positions.platonic(N, ka)

    vertices_dist = []
    for i in range(N):
        for j in range(N):
            if i<j:
                vertices_dist.append(norm(pos[:,i]-pos[:,j]))
    vertices_dist.sort()

    n = int(N*nei/2)
    cond_1 = (np.array([norm(v-valka) for v in vertices_dist[:n]])<1e-10).all()
    if N>n:
        cond_2 = (vertices_dist[n]-valka)>1e-10
    else:
        cond_2 = True

    assert cond_1 & cond_2


def test_platonic_N8():
    N = 8
    nei = 3 # number of neighbors for each vertex
    val = 2/np.sqrt(3) # expected value for unity solid

    ka = np.random.uniform(1,5)
    valka = val*ka
    pos = positions.platonic(N, ka)

    vertices_dist = []
    for i in range(N):
        for j in range(N):
            if i<j:
                vertices_dist.append(norm(pos[:,i]-pos[:,j]))
    vertices_dist.sort()

    n = int(N*nei/2)
    cond_1 = (np.array([norm(v-valka) for v in vertices_dist[:n]])<1e-10).all()
    if N>n:
        cond_2 = (vertices_dist[n]-valka)>1e-10
    else:
        cond_2 = True

    assert cond_1 & cond_2


def test_platonic_N12():
    N = 12
    nei = 5 # number of neighbors for each vertex
    val = 4/np.sqrt(10+2*np.sqrt(5)) # expected value for unity solid

    ka = np.random.uniform(1,5)
    valka = val*ka
    pos = positions.platonic(N, ka)

    vertices_dist = []
    for i in range(N):
        for j in range(N):
            if i<j:
                vertices_dist.append(norm(pos[:,i]-pos[:,j]))
    vertices_dist.sort()

    n = int(N*nei/2)
    cond_1 = (np.array([norm(v-valka) for v in vertices_dist[:n]])<1e-10).all()
    if N>n:
        cond_2 = (vertices_dist[n]-valka)>1e-10
    else:
        cond_2 = True

    assert cond_1 & cond_2


def test_platonic_N20():
    N = 20
    nei = 3 # number of neighbors for each vertex
    val = 4/((1+np.sqrt(5))*np.sqrt(3)) # expected value for unity solid

    ka = np.random.uniform(1,5)
    valka = val*ka
    pos = positions.platonic(N, ka)

    vertices_dist = []
    for i in range(N):
        for j in range(N):
            if i<j:
                vertices_dist.append(norm(pos[:,i]-pos[:,j]))
    vertices_dist.sort()

    n = int(N*nei/2)
    cond_1 = (np.array([norm(v-valka) for v in vertices_dist[:n]])<1e-10).all()
    if N>n:
        cond_2 = (vertices_dist[n]-valka)>1e-10
    else:
        cond_2 = True

    assert cond_1 & cond_2


# test on translations
# --------------------

def test_translations():
    lmax = 4
    n = lmax*(2*lmax+1)

    # prepare extraction of compact indices
    inds = []
    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            ind = (l-1)*(2*lmax+1) + lmax + m
            inds.append(ind)
            inds.append(n+ind)
    inds.sort()
    
    # build random translation and its inverse
    rho = np.random.uniform(0.1, 0.5, 3).reshape(3,1)
    T1 = tools.translate_reduced(rho, lmax)
    T2 = tools.translate_reduced(-rho, lmax)

    # check it is reversible
    P = np.dot(T1, T2)[:,inds][inds,:]
    R = np.identity(P.shape[0])-P
    assert (norm(R)/norm(T1))<0.1


# test for a single dipole
# ------------------------

def test_singledipole():
    lmax = 10
    Nsph = 2*lmax*(2*lmax+1)
    pos = np.reshape(np.array([0, 0, 0]), (3, -1))
    eps = (np.sqrt(10) + 0.1*1j)**2
    alpha_0 = np.array([(eps-1)/(eps+2)])
    alpha = alpha_0 / (1-1j*alpha_0/(6*np.pi))

    D = tools.matrix_Ddip(pos, alpha, lmax)
    
    D_manual = np.zeros((Nsph, Nsph),  dtype=np.complex_)
    v = 1j*alpha[0]/(6*np.pi)
    D_manual[lmax-1, lmax-1] = v
    D_manual[lmax, lmax] = v
    D_manual[lmax+1, lmax+1] = v

    assert (norm(D-D_manual)/norm(D_manual))<1e-10

    






