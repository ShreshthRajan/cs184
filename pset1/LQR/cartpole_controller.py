import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, x, u):
        """
        Cost function of the env.
        It sets the state of environment to `x` and then execute the action `u`, and
        then return the cost. 
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        observation, cost, done, info = env.step(u)
        return cost

    def f(self, x, u):
        """
        State transition function of the environment.
        Return the next state by executing action `u` at the state `x`
        Parameter:
            x (1D numpy array) with shape (4,)
            u (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        next_observation, cost, done, info = env.step(u)
        return next_observation


    def compute_local_policy(self, x_star, u_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (x_star, u_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            x_star (numpy array) with shape (4,)
            u_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        # 1. Compute Jacobians for f
        A = jacobian(lambda x_: self.f(x_, u_star), x_star) # shape (4,4)
        B = jacobian(lambda u_: self.f(x_star, u_), u_star) # shape (4,1)

        # 2. Compute Jacobians and Hessians for c
        cx = gradient(lambda x_: self.c(x_, u_star), x_star) # shape (4,)
        cu = gradient(lambda u_: self.c(x_star, u_), u_star) # shape (1,)
        cxx = hessian(lambda x_: self.c(x_, u_star), x_star) # shape (4,4)
        cuu = hessian(lambda u_: self.c(x_star, u_), u_star) # shape (1,1)

        # Convert gradients to the appropriate shape
        cx = np.reshape(cx, (4, 1))
        cu = np.reshape(cu, (1, 1))

        # 3. Use LQR to compute optimal policy given linearized dynamics and quadratic cost
        # Assuming cost function is approximately quadratic: 0.5 * x^T Q x + 0.5 * u^T R u
        Q = cxx
        R = cuu
        M = np.zeros((4, 1))  # Cross-term, assuming it's zero
        q = cx
        r = cu
        b = np.zeros(1)  # Constant term, assuming it's zero
        m = np.zeros((4, 1))  # Constant term in dynamics, assuming it's zero

        # Call the LQR function to get the optimal policies
        Ks = lqr(A, B, m, Q, R, M, q, r, b, T)

        return Ks

class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a
