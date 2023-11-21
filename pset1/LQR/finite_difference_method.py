import numpy as np


def gradient(f, x, delta=1e-6):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    #TODO
    n, = x.shape
    gradient = np.zeros(n).astype('float64')
    for i in range(n):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += delta
        x2[i] -= delta
        gradient[i] = (f(x1) - f(x2)) / (2*delta)
    return gradient


def jacobian(f, x, delta=1e-6):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    #TODO
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input. 
    gradient = np.zeros((m, n)).astype('float64')
    for i in range(n):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += delta
        x2[i] -= delta
        gradient[:, i] = (f(x1) - f(x2)) / (2*delta)
    return gradient



def hessian(f, x, delta=1e-6):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    #TODO
    n, = x.shape
    gradient = np.zeros((n, n)).astype('float64')
    for i in range(n):
        for j in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x3 = x.copy()
            x4 = x.copy()
            
            x1[i] += delta
            x1[j] += delta
            x2[i] += delta
            x2[j] -= delta
            x3[i] -= delta
            x3[j] += delta
            x4[i] -= delta
            x4[j] -= delta
            
            gradient[i, j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4*delta*delta)
    return gradient
    


