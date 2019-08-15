import numpy as np

def matrix_Sdip(pos, alpha, lmax):
    '''
    generates the scattering matrix S
    from the diffusion matrix D
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size N
    alpha must be already multiplied by k^3 to be adimensional
    '''
    N = pos.shape[1] # number of particles
    D = matrix_Ddip(pos, alpha, lmax)
    S = np.eye(3*N) + 2*D
    return S


def matrix_Ddip(pos, alpha, lmax):
    '''
    generates the diffusion matrix D
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size N
    alpha must be already multiplied by k^3 to be adimensional
    '''
    N = pos.shape[1] # number of particles
    TQ = matrix_TQdip(pos, alpha, lmax) # size Nsph by 3N
    X = matrix_Xdip(pos, alpha) # size 3N by 3N
    FT = matrix_FTdip(pos, lmax) # size 3N by Nsph
    D = np.dot(TQ, np.linalg.inv(np.eye(3*N) - X)).dot(FT)
    return D # size Nsph by Nsph


def matrix_Xdip(pos, alpha):
    '''
    generates the structure X matrix
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size N
    alpha is multiplied by k^3 to be adimensional
    '''

    N = pos.shape[1] # number of particles
    X = np.zeros((3*N, 3*N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                X[3*i:3*(i+1),3*j:3*(j+1)] = alpha[j]*eval_green(pos[:,i],pos[:,j])
    return X



def matrix_FTdip(pos, alpha, lmax):
	'''
    generates the (3N, Nsph) F*T block matrix
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size N
    alpha is multiplied by k^3 to be adimensional
    '''

	N = pos.shape[1] # number of particles
	Nsph = 2*lmax*(2*lmax+1) # number os spherical modes (with reduced notation)

	# construction of the generic F matrix, size 3 by Nsph 
	F = np.zeros((3, Nsph))
	F[:, lmax:lmax+3] = [[1j,   0, -1j]
						 [0,    0, sqrt(2)*j]
						 [-1j, 1, 0]] / np.sqrt(12*np.pi)

	FT = np.zeros((3*n,N))
	for i in range(N):
    	T = trans_j2j_reduced(-pos[:,i],lmax) # size Nsph by Nsph
    	FT[i*3:(i+1)*3,:] = F*T # block of size 3 by Nsph

    return FT # size 3N by Nsph



def matrix_TQdip(pos, alpha, lmax):
    '''
    generates the (Nsph, 3N) T*Q block matrix
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3,N) and alpha has size N
    alpha is multiplied by k^3 to be adimensional
    '''

    N = pos.shape[1] # number of particles
    Nsph = 2*lmax*(2*lmax+1) # number os spherical modes (with reduced notation)

    # construction of the generic Q matrix, size Nsph by 3 
    Q = np.zeros((Nsph, 3))
    Q[lmax:lmax+3,:] = [[1,  1j, 0],
                        [0,  0,  sqrt(2)],
                        [-1, 1j, 0]] / np.sqrt(12*np.pi)
    
    TQ = np.zeros((Nsph,3*N))
    for i in range(N):
        T = trans_h2h_reduced(pos[:,i],lmax) # size Nsph by Nsph
        TQ[:,i*3:(i+1)*3] = alpha[i]*T*Q # size Nsph by 3

    return TQ # size Nsph by 3N




