from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    softmax = exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)
    return softmax



def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """
    logits = np.dot(theta.T, phis)  # shape (1, |A|)
    probabilities = compute_softmax(logits, axis=1)
    return probabilities



def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """

    probabilities = compute_action_distribution(theta, phis)[0]  # shape (|A|,)
    expected_feature_vector = np.dot(phis, probabilities)  # shape (d,)
    
    # Compute the gradient of the log policy
    grad_log_policy = phis[:, action_idx] - expected_feature_vector  # shape (d,)
    return grad_log_policy.reshape(-1, 1)



def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the Fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)
    """
    d = len(grads[0][0])  # Assuming all gradients have the same dimension
    fisher_matrix = np.zeros((d, d))

    total_grads = 0
    for grad_trajectory in grads:  # Iterate over trajectories
        for grad in grad_trajectory:  # Iterate over gradients within a trajectory
            fisher_matrix += np.outer(grad, grad)
            total_grads += 1

    # Normalize by the total number of state-action pairs
    fisher_matrix /= total_grads

    # Add regularization
    fisher_matrix += lamb * np.eye(d)

    return fisher_matrix


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: list of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """
    # Calculate baseline: average of the sum of rewards for each trajectory
    baseline = np.mean([np.sum(traj_rewards) for traj_rewards in rewards])

    # Assuming all gradients have the same dimension
    d = len(grads[0][0])
    value_grad = np.zeros((d, 1))

    # Loop over each trajectory
    for grad_trajectory, reward_trajectory in zip(grads, rewards):
        for h, grad in enumerate(grad_trajectory):
            # Sum of rewards from step h to the end of the trajectory
            cumulative_reward = np.sum(reward_trajectory[h:]) - baseline
            # Update value function gradient
            value_grad += grad * cumulative_reward

    # Normalize by the number of trajectories
    value_grad /= len(grads)

    return value_grad

def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """
    epsilon = 1e-6  # Small constant to avoid division by zero

    # Ensure the Fisher matrix is invertible; handle potential numerical issues
    fisher_inv = np.linalg.pinv(fisher)  # Using pseudo-inverse for numerical stability

    # Compute the quadratic form
    quadratic_form = v_grad.T @ fisher_inv @ v_grad

    # Ensure the result of the quadratic form is non-negative
    quadratic_form_value = max(quadratic_form.item(), 0)

    # Compute eta
    eta = np.sqrt(delta / (quadratic_form_value + epsilon))

    return eta


