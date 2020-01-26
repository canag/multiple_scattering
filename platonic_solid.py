import numpy as np

def platonic(N, ka):
    if N==4:
        return tetrahedron(ka)
    elif N==6:
        return octahedron(ka)
    elif N==8:
        return cube(ka)
    elif N==12:
        return icosahedron(ka)
    elif N==20:
        return dodecahedron(ka)
    else:
        print("Issue with N parameter")
        pass


def tetrahedron(ka):
    pos = np.zeros((3,4))

    pos[:,0] = [ 1, 1, 1]
    pos[:,1] = [ 1,-1,-1]
    pos[:,2] = [-1, 1,-1]
    pos[:,3] = [-1,-1, 1]

    return pos*ka/np.sqrt(3)


def octahedron(ka):
    pos = np.zeros((3,6))

    pos[:,0] = [ 1, 0, 0]
    pos[:,1] = [-1, 0, 0]
    pos[:,2] = [ 0, 1, 0]
    pos[:,3] = [ 0,-1, 0]
    pos[:,4] = [ 0, 0, 1]
    pos[:,5] = [ 0, 0,-1]

    return pos*ka

def cube(ka):
    pos = np.zeros((3,8))

    pos[:,0] = [ 1, 1, 1]
    pos[:,1] = [ 1, 1, -1]
    pos[:,2] = [ 1,-1, 1]
    pos[:,3] = [ 1,-1,-1]
    pos[:,4] = [-1, 1, 1]
    pos[:,5] = [-1, 1, -1]
    pos[:,6] = [-1,-1, 1]
    pos[:,7] = [-1,-1,-1]

    return pos*ka/np.sqrt(3)


def icosahedron(ka):
    pos = np.zeros((3,12))
    phi = (1+np.sqrt(5)) / 2

    pos[:, 0] = [ 1, 0, phi]
    pos[:, 1] = [-1, 0, phi]
    pos[:, 2] = [ 1, 0,-phi]
    pos[:, 3] = [-1, 0,-phi]
    
    pos[:, 4] = [ phi, 1, 0]
    pos[:, 5] = [ phi,-1, 0]
    pos[:, 6] = [-phi, 1, 0]
    pos[:, 7] = [-phi,-1, 0]

    pos[:, 8] = [0, phi, 1]
    pos[:, 9] = [0, phi,-1]
    pos[:,10] = [0,-phi, 1]
    pos[:,11] = [0,-phi, 1]   

    return pos*ka/np.sqrt(1+phi**2)

def dodecahedron(ka):
    pos = np.zeros((3,20))
    phi = (1+np.sqrt(5)) / 2

    pos[:, 0] = [0, 1/phi, phi]
    pos[:, 1] = [0, 1/phi,-phi]
    pos[:, 2] = [0,-1/phi, phi]
    pos[:, 3] = [0,-1/phi,-phi]

    pos[:, 4] = [ 1/phi, phi, 0]
    pos[:, 5] = [ 1/phi,-phi, 0]
    pos[:, 6] = [-1/phi, phi, 0]
    pos[:, 7] = [-1/phi,-phi, 0]

    pos[:, 8] = [ phi, 0, 1/phi]
    pos[:, 9] = [ phi, 0,-1/phi]
    pos[:,10] = [-phi, 0, 1/phi]
    pos[:,11] = [-phi, 0,-1/phi]

    pos[:,12] = [ 1, 1, 1]
    pos[:,13] = [ 1, 1,-1]
    pos[:,14] = [ 1,-1, 1]
    pos[:,15] = [ 1,-1,-1]
    pos[:,16] = [-1, 1, 1]
    pos[:,17] = [-1, 1,-1]
    pos[:,18] = [-1,-1, 1]
    pos[:,19] = [-1,-1,-1]
 
    return pos*ka/np.sqrt(3)