import pytest 
import numpy as np
import np.linalg.norm as norm2
import collective_Smatrix as tools
import platonic_solid as positions

def test_CGcoeff1010():
    a = np.array([-1/np.sqrt(3), 0, np.sqrt(2/3), 0])
    b = tools.CG_coeff(1, 0, 1, 0)
    assert norm2(a-b)/norm2(a)<1e-10
