import numpy as np


# first level
# -----------

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


# second level
# ------------

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
    	T = translate_reduced(-pos[:,i], lmax) # size Nsph by Nsph
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
        T = translate_reduced(pos[:,i], lmax) # size Nsph by Nsph
        TQ[:,i*3:(i+1)*3] = alpha[i]*T*Q # size Nsph by 3

    return TQ # size Nsph by 3N


# third level
# -----------

def translate_reduced(rho, lmax):
	'''
	function that builds the translation matrix of size Nsph by Nsph
	for both j2j and h2h modes, with reduced motation
	rho is of size 3
	'''
	n = lmax*(2*lmax+1)
	Nsph = 2*n

	A = np.zeros((Nsph, Nsph))
	if np.abs(rho)==0:
		A = np.ones((Nsph, Nsph))
	else
		B = Brho_matrix(rho, lmax) # size n by n
		C = Crho_matrix(rho, lmax) # idem
		A[:n, :n] = B
		A[:n, n:] = 1j*C
		A[n:, :n] = -1j*C
		A[n:, n:] = B

	return A



def Brho_matrix(rho, lmax):
	''' 
	derives the matrix B that applies translations for vectorial spherical modes
	B is the diagonal block (M to M and N to N) of size n by n where n = Nsph/2
	rho=k*(rho_x,rho_y,rho_z) is the translation vector times k
	'''

	# finite number of modes in vectorial spherical modes
	n = lmax*(2*lmax+1)
	# indexing for both l and m: i = (l-1)*(2*lmax+1)+m+lmax
	# 1<=l<=lmax and -lmax<=m<=lmax gives i in [0,n-1]
	B = np.zeros(n, n)

	# scalar spherical harmonics evaluated at rho
	# linevector of size n_u = (2*lmax+1)*(4*lmax+1)
	# with index ind = alpha*(4*lmax+1)+beta+2*lmax
	# where alpha from 0 to 2*lmax
	# and beta from -2*lmax to +2*lmax
	# gives ind from 0 to n_u-1
	u_rho = eval_u1(2*lmax, rho) 

	for l1 in range(1, lmax+1):
	    for l2 in range(1, lmax+1):
	        for m1 in range(-l1, l1+1):
	            for m2 in range(-l2, l2+1):
	                i1 = (l1-1)*(2*lmax+1) + m1 + lmax
	                i2 = (l2-1)*(2*lmax+1) + m2 + lmax
	                
	                alpha = np.arange(0, l1+l2+1) # size l1+l2+1
	                a = a_coeff(l1,-m1,l2,m2) # 0<=alpha<=l1+l2, size l1+l2+1
	                K_alpha = (l1*(l1+1)+l2*(l2+1)-alpha*(alpha+1)) / (2*np.sqrt(l1*(l1+1)*l2*(l2+1)))

	                ind = alpha*(4*lmax+1)+m2-m1+2*lmax # indices where beta=m2-m1
	                u = u_rho[ind]

	                # sum for alpha from 0 to l1+l2
	                B[i1, i2] = ((-1)^m1) * 4*np.pi*1i^(l2-l1) * np.sum(1i^alpha*K_alpha*a*u)
	            
	return B



def Crho_matrix(rho, lmax):
	''' 
	derives the matrix C that applies translations for vectorial spherical modes
	C is the non-diagonal block (M to N and N to M) of size n by n where n = Nsph/2
	rho=k*(rho_x,rho_y,rho_z) is the translation vector times k
	'''

	# finite number of modes in vectorial spherical modes
	n = lmax*(2*lmax+1)
	# indexing for both l and m: i = (l-1)*(2*lmax+1)+m+lmax
	# 1<=l<=lmax and -lmax<=m<=lmax gives i in [0,n-1]
	C = np.zeros((n, n))

	# square matrix of size (lmax+2)*(2*lmax+3) by (lmax+2)*(2*lmax+3)
	# where l from 0 to lmax+1
	# and m from -(lmax+1) to +(lmax+1)
	# with a corresponding indexing is iA = l*(2*lmax+3)+m+lmax+1
	A = Arho_matrix(rho, lmax+1) # size (lmax+2)*(2*lmax+3) by (lmax+2)*(2*lmax+3)

	for l1 in range(1, lmax+1):
		for l2 in range(1, lmax+1):
			K = -1j/np.sqrt(l1*(l1+1)*l2*(l2+1)) # scalar
		    for m1 in range(-l1, l1+1):
		        for m2 in range(-l2, l2+1):
		        	# indices for C matrix
		            i1 = (l1-1)*(2*lmax+1) + m1 + lmax
		            i2 = (l2-1)*(2*lmax+1) + m2 + lmax
		            # indices for A matrix
		            iA1 = l1*(2*lmax+3) + m1 + lmax+1
                	iA2 = l2*(2*lmax+3) + m2 + lmax+1

                	lambda_p = np.sqrt((l2-m2)*(l2+m2+1))
                	lambda_m = np.sqrt((l2+m2)*(l2-m2+1))
		            
		            Cx = (lambda_p*A[iA1,iA2+1] + lambda_m*A[iA1,iA2-1]) / 2
                	Cy = (lambda_p*A[iA1,iA2+1] - lambda_m*A[iA1,iA2-1]) / (2*1j)
                	Cz = m2*A[iA1,iA2]
                	C[i1,i2] = K*(rho[0]*Cx + rho[1]*Cy + rho[2]*Cz)

	return C


def eval_green


# fourth level
# ------------

def Arho_matrix
def eval_u1
def a_coeff
