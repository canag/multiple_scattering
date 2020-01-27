
import numpy as np

def integrand_NP_imagaxis(akappa, aGamma, ak0, delta, pos):
    '''
    function that evaluates the integrand on the imaginary axis
    from the X operator evaluated for xi = i*omega
    akappa can be a vector of dimension n
    delta, aGamma and ak0 are scalar
    the 2 later correspond to adimensional a*Gamma/c and a*omega_0/c
    pos has size (3,N) and is normalized by ka
    '''

    N = pos.shape[1] # number of particles
    z = 0*akappa # vector of size n

    for i, ak in enumerate(akappa):
        alpha = -6*np.pi*aGamma*ak**3 / (ak0**2*(ak**2 + ak0**2 + np.sqrt(3)*delta*ak*ak0) - aGamma*ak**3)
        X = matrix_Xdip_xi(pos*ak, alpha + np.zeros(N))
        z[i] = np.log(np.linalg.det(np.identity(3*N) - X))

    return z


def matrix_Xdip_xi(pos, alpha):
    '''
    generates the structure X matrix at imaginary frequencies
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
                X[(3*i):(3*(i+1)),(3*j):(3*(j+1))] = alpha[j] * eval_green_xi(pos[:,i],pos[:,j])

    return X


def eval_green_xi(pos1, pos2):
    '''
    function that evaluates the Green tensor in vacuum 
    at imaginary frequencies, divided by kappa
    between positions 1 and 2 (numpy vectors of size 3) written in units of k
    and returns a 3 by 3 complex matrix
    '''
	
    R = np.linalg.norm(pos1-pos2, 2) # scalar
    u = (pos1-pos2)/R # unit vector, dim 3 by 1

    M = np.dot(u.reshape(-1,1), u.reshape(1, -1)) # 3 by 3 matrix
    G = np.exp(-R) * ((R**2 + R + 1)/R**2*np.eye(3)
                      - (R**2 + 3*R + 3)/R**2*M) / (4*np.pi*R)
    
    return G