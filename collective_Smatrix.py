import numpy as np
from math import factorial as fact
from scipy.special import jv, lpmv


# first level
# -----------


def matrix_Sdip(pos, alpha, lmax):
    '''
    generates the scattering matrix S
    with spherical modes up to l=lmax
    in the case of N dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3, N) and alpha has size N
    alpha is multiplied by k**3 to be adimensional
    '''
    
    # number of spherical modes
    Nsph = 2*lmax*(2*lmax+1)
    
    # diffusion matrix for the N scatterers
    D = matrix_Ddip(pos, alpha, lmax)
    
    # scattering matrix for the N scatterers
    S = np.eye(Nsph) + 2*D
    
    return S


def matrix_Ddip(pos, alpha, lmax):
    '''
    generates the diffusion matrix D
    with spherical modes up to l=lmax
    in the case of N dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3, N) and alpha has size N
    alpha is multiplied by k**3 to be adimensional
    '''
    
    N = pos.shape[1] # number of particles
    
    TQ = matrix_TQdip(pos, alpha, lmax) # size Nsph by 3N
    X = matrix_Xdip(pos, alpha) # size 3N by 3N
    FT = matrix_FTdip(pos, alpha, lmax) # size 3N by Nsph
    
    D = np.dot(TQ, np.linalg.inv(np.eye(3*N) - X)).dot(FT)

    return D # size Nsph by Nsph


### subfunctions needed for first level: matrix_TQdip, matrix_Xdip, matrix_FTdip


# second level
# ------------


def matrix_Xdip(pos, alpha):
    '''
    generates the structure X matrix
    in the case of N dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3, N) and alpha has size N
    alpha is multiplied by k**3 to be adimensional
    '''

    N = pos.shape[1] # number of particles
    X = np.zeros((3*N, 3*N),  dtype=np.complex_)

    for i in range(N):
        for j in range(N):
            if i!=j: # outside diagonal blocks
                X[(3*i):(3*(i+1)),(3*j):(3*(j+1))] = alpha[j] * eval_green(pos[:,i],pos[:,j])

    return X


def matrix_FTdip(pos, alpha, lmax):
    '''
    generates the (3*N, Nsph) F*T block matrix
    in the case of N dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3, N) and alpha has size N
    alpha is multiplied by k**3 to be adimensional
    '''

    N = pos.shape[1] # number of particles
    Nsph = 2*lmax*(2*lmax+1) # number of spherical modes (with reduced notation)

    # conversion between double (l,m) and simple (i) indexing:
    # l from 1 to lmax and m from -l to l (-> -lmax, +lmax for simplicity)
    # i = (l-1)*(2*lmax+1)+m+lmax goes from 0 to Nsph/2-1

    # construction of the generic F matrix, size 3 by Nsph 
    F = np.zeros((3, Nsph), dtype=np.complex_)
    # only non-zero rows are Alm with l=1, m from -1 to 1
    # then i from lmax-1 to lmax+1 (included)
    F[:, (lmax-1):(lmax+2)] = [[1j,   0, -1j],
                               [1,    0, 1], 
                               [0,  1j*np.sqrt(2), 0]] / np.sqrt(12*np.pi)

    FT = np.zeros((3*N, Nsph), dtype=np.complex_)
    for i in range(N):
        T = translate_reduced(-pos[:,i], lmax) # size Nsph by Nsph
        FT[(i*3):((i+1)*3),:] = np.dot(F,T) # block of size 3 by Nsph
    
    return FT # size 3N by Nsph


def matrix_TQdip(pos, alpha, lmax):
    '''
    generates the (Nsph, 3N) T*Q block matrix
    in the case of dipolar scatterers
    with positions pos and polarizabilities alpha
    pos has size (3, N) and alpha has size N
    alpha is multiplied by k**3 to be adimensional
    '''

    N = pos.shape[1] # number of particles
    Nsph = 2*lmax*(2*lmax+1) # number os spherical modes (with reduced notation)
    # l from 1 to lmax and m from -l to l (-> -lmax, +lmax)
    # i = (l-1)*(2*lmax+1)+m+lmax

    # construction of the generic Q matrix, size Nsph by 3 
    Q = np.zeros((Nsph, 3), dtype=np.complex_)
    # only non-zero rows are Alm with l=1 m from -1 to 1
    # then i from lmax-1 to lmax+1 (included)
    Q[(lmax-1):(lmax+2),:] = [[1,  1j, 0],
                              [0,  0,  np.sqrt(2)],
                              [-1, 1j, 0]] / np.sqrt(12*np.pi)

    TQ = np.zeros((Nsph,3*N), dtype=np.complex_)
    for i in range(N):
        T = translate_reduced(pos[:,i], lmax) # size Nsph by Nsph
        TQ[:, 3*i : 3*(i+1)] = alpha[i] * np.dot(T,Q) # size Nsph by 3

    return TQ # size Nsph by 3N


### subfunctions needed for second level: eval_green, translate_reduced


# third level
# -----------


def eval_green(pos1, pos2):
    '''
    function that evaluates the Green tensor in vacuum divided by k
    between positions 1 and 2 (numpy vectors of size 3) written in units of k
    and returns a 3 by 3 complex matrix
    '''
	
    R = np.linalg.norm(pos1-pos2, 2) # scalar
    u = (pos1-pos2)/R # unit vector, dim 3 by 1

    M = np.dot(u.reshape(-1,1), u.reshape(1, -1)) # 3 by 3 matrix
    G = np.exp(1j*R) * ((R**2 + 1j*R - 1)/R**2*np.eye(3)
                        - (R**2 + 3*1j*R - 3)/R**2*M) / (4*np.pi*R)
    
    return G


def translate_reduced(rho, lmax):
    '''
    function that builds the translation matrix of size Nsph by Nsph
    for both j2j and h2h modes, with reduced motation
    rho is of size 3
    '''
    
    # n: size of the 4 square-blocks composing T matrix
    n = lmax*(2*lmax+1)  
    Nsph = 2*n 

    T = np.zeros((Nsph, Nsph), dtype=np.complex_)
    
    if np.linalg.norm(rho)==0: # otherwise unstable behavior
        T = T + np.ones((Nsph, Nsph)) # size 2n by 2n
    else:
        B = Brho_matrix(rho, lmax) # size n by n
        C = Crho_matrix(rho, lmax) # idem
        T[:n, :n] = B
        T[:n, n:] = 1j*C
        T[n:, :n] = -1j*C
        T[n:, n:] = B

    return T


### subfunctions needed for third level: Brho_matrix, Crho_matrix


# fourth level
# -----------


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
    B = np.zeros((n, n), dtype=np.complex_)

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
                    B[i1, i2] = ((-1)**m1) * 4*np.pi*1j**(l2-l1) * np.sum(1j**alpha*K_alpha*a*u)

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
    C = np.zeros((n, n), dtype=np.complex_)

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


def Arho_matrix(rho, lmax):
    '''
    function that derives the matrix A that applies translations for scalar 
    spherical modes, rho=k*(rho_x,rho_y,rho_z) is the translation vector times k
    '''

    # finite number of modes in scalar spherical modes
    Nsca = (lmax+1)*(2*lmax+1)
    # i = l*(2*lmax+1) + m + lmax
    # 0<=l<=lmax and -lmax<=m<=lmax
    A = np.zeros((Nsca, Nsca), dtype=np.complex_)
    
    # spherical harmonics evaluated at rho
    # linevector of size (2*lmax+1)*(4*lmax+1)
    # with index i = alpha*(4*lmax+1) + beta + 2*lmax
    u_rho = eval_u1(2*lmax, rho)

    for l1 in range(lmax+1): # from 0 to lmax 
        for l2 in range(lmax+1): # from 0 to lmax
            for m1 in range(-l1, l1+1): # from -l1 to l1
                for m2 in range(-l2, l2+1): # from -l2 to l2
                    i1 = l1*(2*lmax+1) + m1 + lmax
                    i2 = l2*(2*lmax+1) + m2 + lmax
                    
                    alpha = np.arange(l1+l2+1) # from 0 to (l1+l2), size l1+l2+1
                    a = a_coeff(l1, -m1, l2, m2) # 0<=alpha<=l1+l2

                    # indices where beta=m2-m1
                    ind = alpha*(4*lmax+1)+m2-m1+2*lmax
                    u = u_rho[ind]
                    
                    # sum for alpha from 0 to l1+l2
                    A[i1,i2] = (-1)**m1 * 4*np.pi*1j**(l2-l1) * np.sum(1j**alpha*a*u)
    return A


### subfunctions needed for fourth level: eval_u1, a_coeff


# fifth level
# ------------

def eval_u1(lmax, pos):
	'''
	function that evaluates the spherical modes (in j) up to lmax
	for position pos=(x,y,z)*k (scalar quantity)
	and return u, numpy complex vector of size (lmax+1)*(2*lmax+1)
	for 0 <= l <= lmax and -l <= m <= l 
	'''
	
	kr = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
	costheta = pos[2]/kr
	if (pos[0]**2+pos[1]**2)>0:
		eiphi = (pos[0]+1j*pos[1]) / np.sqrt(pos[0]**2+pos[1]**2)
	else: # theta=0 and phi is undetermined
		eiphi = 1
	
	# when costheta=1, Plm is zero except for m=0 where it is 1
	# hence eiphi only appears for m=0 where eiphi^m=1

	u = np.zeros((lmax+1)*(2*lmax+1), dtype=np.complex_)

	for l in range(lmax+1): # 0<=l<=lmax
		# computes for all values 0<=m<=l
		P = np.zeros(l+1)
		for m in range(l+1):
			P[m] = lpmv(m, l , costheta)	
		Y = np.zeros(2*lmax+1, dtype=np.complex_)

		# m from -lmax to lmax
		# only cases with m from -l to l will be computed 
		for m in range(l+1): # 0<=m<=l
			# i = m+lmax+1
			Klm = np.sqrt((2*l+1)*fact(l-m)/(4*np.pi*fact(l+m)))
			Y[m+lmax] = Klm * P[m] * eiphi**m
			Y[-m+lmax] = Klm * P[m] * eiphi**(-m) * (-1)**m
		
		jl = np.sqrt(np.pi/(2*kr))*jv(l+1/2, kr) # scalar
		u[l*(2*lmax+1):((l+1)*(2*lmax+1))] = jl*Y # size 2*lmax+1

	return u 


def a_coeff(l1, m1, l2, m2):
	'''
	function that computes the coeff a(alpha,beta) that corresponds
	to the decomposition of the product of Y's in spherical harmonics
	0<=alpha<=l1+l2 and beta=m1+m2
	and returns a numpy vector of size l1+l2+1
	'''

	N_alpha = l1+l2+1
	alpha = np.arange(N_alpha)
	C = CG_coeff(l1, m1, l2, m2) # size l1+l2+2 for alpha from 0 to l1+l2+1
	C0 = CG_coeff(l1, 0, l2, 0) # idem
	# only uses values of alpha from 0 to l1+l2 (not the last one)
	a = np.sqrt((2*l1+1)*(2*l2+1)/(4*np.pi*(2*alpha+1))) * C[:N_alpha] * C0[:N_alpha]
	return a


def CG_coeff(l1, m1, l2, m2):
	'''
	function that derives the Clebsch-Gordan coefficients C
	for 0<=alpha<=l1+l2+1 and beta=m1+m2
	and returns a numpy vector of size l1+l2+2
	'''
	
	beta = m1 + m2
	N = l1 + l2 + 2
	C = np.zeros(N)

	alpha=l1+l2+1 # (=N-1, and i=l1+l2+2=N)
	C[alpha] = 0

	alpha = l1+l2 # (=N-2, and i=l1+l2+1=N-1)
	C[alpha] = np.sqrt(fact(l1+l2+m1+m2) * fact(l1+l2-m1-m2)/(fact(l1+m1) * fact(l1-m1) * fact(l2+m2) * fact(l2-m2))) * np.sqrt(fact(2*l1) * fact(2*l2) / fact(2*l1+2*l2))
	
	for alpha in range(N-3,-1,-1):
		if (alpha >= max(abs(l2-l1),abs(beta))):
			zeta = m1 - m2 - beta*(l1*(l1+1)-l2*(l2+1))/((alpha+1)*(alpha+2))
			xi = ((alpha+1)**2-beta**2) * ((alpha+1)**2-(l1-l2)**2) * ((l1+l2+1)**2-(alpha+1)**2)/((alpha+1)**2*(4*(alpha+1)**2-1))
			xiplus = ((alpha+2)**2-beta^2) * ((alpha+2)**2-(l1-l2)**2) * ((l1+l2+1)**2-(alpha+2)**2)/((alpha+2)**2*(4*(alpha+2)**2-1))
			C[alpha] = zeta/np.sqrt(xi)*C[alpha+1] - np.sqrt(xiplus/xi)*C[alpha+2]

	return C
