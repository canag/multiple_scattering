import numpy as np

def matrix_Ddip(pos,alpha,lmax):
    '''generates the scattering matrix D
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size (1,N)
    alpha must be already multiplied by k^3 to be adimensional
    '''

    N = pos.shape[1]
    TQ = matrix_TQdip(pos,alpha,lmax) # size Nsph by 3N
    X = matrix_Xdip(pos,alpha) # size 3N by 3N
    FT = matrix_FTdip(pos,lmax) # size 3N by Nsph
    D = np.dot(TQ, np.linalg.inv(np.eye(3*N) - X)).dot(FT)
    return D # size Nsph by Nsph
