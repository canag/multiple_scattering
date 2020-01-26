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
