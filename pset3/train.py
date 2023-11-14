import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    trajectories_gradients = []
    trajectories_rewards = []

    for _ in range(N):
        state = env.reset()
        rewards = []
        gradients = []
        for _ in range(200):  # Maximum trajectory length is 200 steps
            phis = utils.extract_features(state, env.action_space.n)
            action_probs = utils.compute_action_distribution(theta, phis)
            action = np.random.choice(env.action_space.n, p=action_probs.flatten())

            next_state, reward, done, _ = env.step(action)
            grad_log_policy = utils.compute_log_softmax_grad(theta, phis, action)

            rewards.append(reward)
            gradients.append(grad_log_policy)

            state = next_state
            if done:
                break

        trajectories_rewards.append(rewards)
        trajectories_gradients.append(gradients)

    return trajectories_gradients, trajectories_rewards



import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    trajectories_gradients = []
    trajectories_rewards = []

    for _ in range(N):
        state = env.reset()
        rewards = []
        gradients = []
        for _ in range(200):  # Maximum trajectory length is 200 steps
            phis = utils.extract_features(state, env.action_space.n)
            action_probs = utils.compute_action_distribution(theta, phis)
            action = np.random.choice(env.action_space.n, p=action_probs.flatten())

            next_state, reward, done, _ = env.step(action)
            grad_log_policy = utils.compute_log_softmax_grad(theta, phis, action)

            rewards.append(reward)
            gradients.append(grad_log_policy)

            state = next_state
            if done:
                break

        trajectories_rewards.append(rewards)
        trajectories_gradients.append(gradients)

    return trajectories_gradients, trajectories_rewards



def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100, 1)  # Random initialization of theta
    env = gym.make('CartPole-v0')
    env.seed(12345)

    episode_rewards = []

    for _ in range(T):  # Iterate for T training steps
        # Collect samples
        gradients, rewards = sample(theta, env, N)

        # Compute Fisher Information matrix
        fisher_matrix = utils.compute_fisher_matrix(gradients, lamb)

        # Compute the value function gradient
        value_grad = utils.compute_value_gradient(gradients, rewards)

        # Compute the step size
        eta = utils.compute_eta(delta, fisher_matrix, value_grad)

        # Update the model parameters (theta)
        theta += eta * np.linalg.inv(fisher_matrix) @ value_grad

        # Calculate average reward for this iteration and append to episode_rewards
        avg_reward = np.mean([np.sum(traj_rewards) for traj_rewards in rewards])
        episode_rewards.append(avg_reward)

    return theta, episode_rewards

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()


if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
