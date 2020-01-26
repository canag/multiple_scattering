
import numpy as np

def integrand_imag_cutoff(akappa, aGamma, ak0, arho, pos):
    '''
    function that evaluates the integrand on the imaginary axis
    from the X operator evaluated for xi = i*omega
    akappa can be a vector of dimension n
    aGamma, ak0 and arho are scalar
    that corrrespond to adimensional a*Gamma/c, a*omega_0/c and a*rho/c
    pos has size (3,N) and is normalized by ka
    '''

    N = pos.shape[1] # number of particles

    # alpha is a vector of size n, like akappa
    alpha = -6*pi*aGamma*akappa**3 / (ak0**2*(akappa**2+ak0**2)
            - aGamma*akappa**3 / (1+akappa/arho))
    
    z = 0*akappa # vector of size n

    for i, ak in enumerate(akappa):
        alpha = -6*pi*aGamma*ak**3 / (ak0**2*(ak**2+ak0**2)
                                      - aGamma*ak**3 / (1+ak/arho))
        X = matrix_Xdip_xi(pos*ak, alpha)
        z[i] = np.log(np.linalg.det(np.identity(3*N) - X))

    return z


