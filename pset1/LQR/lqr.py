import numpy as np


def compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p):
    """
    Compute the Q function parameters for time step t.
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
        Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        Q (2d numpy array): A numpy array with shape (n_x, n_x). Q is PD
        R (2d numpy array): A numpy array with shape (n_u, n_u). R is PD.
        M (2d numpy array): A numpy array with shape (n_x, n_u)
        q (2d numpy array): A numpy array with shape (n_x, 1)
        r (2d numpy array): A numpy array with shape (n_u, 1)
        b (1d numpy array): A numpy array with shape (1,)
        P (2d numpy array): A numpy array with shape (n_x, n_x). This is the quadratic term of the
            value function equation from time step t+1. P is PSD.
        y (2d numpy array): A numpy array with shape (n_x, 1).  This is the linear term
            of the value function equation from time step t+1
        p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
            value function equation from time step t+1
    Returns:
        C (2d numpy array): A numpy array with shape (n_x, n_x)
        D (2d numpy array): A numpy array with shape (n_u, n_u)
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x,1)
        g (2d numpy array): A numpy array with shape (n_u,1)
        h (1d numpy array): A numpy array with shape (1,)

        where the following equation should hold
        Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a + h

    """
    # TODO
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert Q.shape == (n_x, n_x)
    assert R.shape == (n_u, n_u)
    assert M.shape == (n_x, n_u)
    assert q.shape == (n_x, 1)
    assert r.shape == (n_u, 1)
    assert b.shape == (1, )
    assert P.shape == (n_x, n_x)
    assert y.shape == (n_x, 1)
    assert p.shape == (1, )

    C = np.zeros((n_x, n_x))
    D = np.zeros((n_x, n_u))
    E = np.zeros((n_x, n_u))
    f = np.zeros(n_x, 1)
    g = np.zeros(n_u, 1)
    h = np.zeros(1)

    C = Q + A.T @ P @ A
    D = R + B.T @ P @ B
    E = M + A.T @ P @ B
    f = q + A.T @ (P @ m + y)
    g = r + B.T @ (P @ m + y)
    h = b + 0.5 * m.T @ P @ m + m.T @ y + p

    return C, D, E, f, g, h


def compute_policy(A, B, m, C, D, E, f, g, h):
    """
    Compute the optimal policy at the current time step t
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)


    Let Q_t^*(x) = x^T C x + u^T D u + x^T E u + f^T x  + g^T u  + h
    Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        C (2d numpy array): A numpy array with shape (n_x, n_x). C is PD.
        D (2d numpy array): A numpy array with shape (n_u, n_u). D is PD.
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x, 1)
        g (2d numpy array): A numpy array with shape (n_u, 1)
        h (1d numpy array): A numpy array with shape (1, )
    Returns:
        K_t (2d numpy array): A numpy array with shape (n_u, n_x)
        k_t (2d numpy array): A numpy array with shape (n_u, 1)

        where the following holds
        \pi*_t(x) = K_t x + k_t
    """
    # TODO
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert C.shape == (n_x, n_x)
    assert D.shape == (n_u, n_u)
    assert E.shape == (n_x, n_u)
    assert f.shape == (n_x, 1)
    assert g.shape == (n_u, 1)
    assert h.shape == (1, )

    K_t = np.zeros((n_u, n_x))
    k_t = np.zeros(n_u)

    inv_D = np.linalg.inv(D)
    K_t = -inv_D @ (E.T + B.T @ C)
    k_t = -inv_D @ g

    return K_t, k_t


def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    """
    Compute the V function parameters for the next time step
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
    Let V_t^*(x) = x^TP_tx + y_t^Tx + p_t
    Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        C (2d numpy array): A numpy array with shape (n_x, n_x). C is PD.
        D (2d numpy array): A numpy array with shape (n_u, n_u). D is PD.
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x, 1)
        g (2d numpy array): A numpy array with shape (n_u, 1)
        h (1d numpy array): A numpy array with shape (1, )
        K (2d numpy array): A numpy array with shape (n_u, n_x)
        k (2d numpy array): A numpy array with shape (n_u, 1)

    Returns:
        P_h (2d numpy array): A numpy array with shape (n_x, n_x)
        y_h (2d numpy array): A numpy array with shape (n_x, 1)
        p_h (1d numpy array): A numpy array with shape (1,)
    """
    # TODO
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert C.shape == (n_x, n_x)
    assert D.shape == (n_u, n_u)
    assert E.shape == (n_x, n_u)
    assert f.shape == (n_x, 1)
    assert g.shape == (n_u, 1)
    assert h.shape == (1, )
    assert K.shape == (n_u, n_x)
    assert k.shape == (n_u, 1)

    P_h = np.zeros((n_x, n_x))
    y_h = np.zeros((n_x, 1))
    p_h = np.zeros(1)

    P_h = C + K.T @ D @ K + (K.T @ E + E.T @ K)
    y_h = f + K.T @ D @ k + K.T @ g + E.T @ k
    p_h = h + 0.5 * k.T @ D @ k + k.T @ g

    return P_h, y_h, p_h


def lqr(A, B, m, Q, R, M, q, r, b, T):
    """
    Compute optimal policies by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} x_t^T Q x_t + u_t^T R u_t + x_t^T M u_t + q^T x_t + r^T u_t
    subject to x_{t+1} = A x_t + B u_t + m, u_t = \pi_t(x_t)

    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
    Let optimal \pi*_t(x) = K_t x + k_t

    Parameters:
    A (2d numpy array): A numpy array with shape (n_x, n_x)
    B (2d numpy array): A numpy array with shape (n_x, n_u)
    m (2d numpy array): A numpy array with shape (n_x, 1)
    Q (2d numpy array): A numpy array with shape (n_x, n_x). Q is PD.
    R (2d numpy array): A numpy array with shape (n_u, n_u). R is PD.
    M (2d numpy array): A numpy array with shape (n_x, n_u)
    q (2d numpy array): A numpy array with shape (n_x, 1)
    r (2d numpy array): A numpy array with shape (n_u, 1)
    b (1d numpy array): A numpy array with shape (1,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (n_u, n_x), the shape of k_t is (n_u,)
    """
    # TODO
    n_x, n_u = B.shape

    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert Q.shape == (n_x, n_x)
    assert R.shape == (n_u, n_u)
    assert M.shape == (n_x, n_u)
    assert q.shape == (n_x, 1)
    assert r.shape == (n_u, 1)
    assert b.shape == (1, )

    P, y, p = Q, q, b
    ret = []

    for t in range(T - 1, -1, -1):  # Backward iteration
        C, D, E, f, g, h = compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p)
        K, k = compute_policy(A, B, m, C, D, E, f, g, h)
        P, y, p = compute_V_params(A, B, m, C, D, E, f, g, h, K, k)

        ret.insert(0, (K, k))
        
    return [(np.zeros((n_u, n_x)), np.zeros(n_u)) for _ in range(T)]