from env_MAB import *
from functools import lru_cache
from scipy.stats import beta
import numpy as np
import math


def random_argmax(a):
    '''
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    '''
    return np.random.choice(np.where(a == a.max())[0])

class Explore():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        # Get the record of pulls for each arm
        record = self.MAB.get_record()
        # Calculate the total number of pulls for each arm
        total_pulls = record.sum(axis=1)
        # Find the arm with the minimum number of pulls
        min_pulls = np.min(total_pulls)
        arms_with_min_pulls = np.where(total_pulls == min_pulls)[0]
        # Randomly select an arm among those with the minimum pulls
        selected_arm = np.random.choice(arms_with_min_pulls)
        # Pull the selected arm
        self.MAB.pull(selected_arm)

class Greedy():
    def __init__(self, MAB):
        self.MAB = MAB
        self.initial_pulls = 0  # To track if all arms are pulled once

    def reset(self):
        self.MAB.reset()
        self.initial_pulls = 0

    def play_one_step(self):
        K = self.MAB.get_K()

        # Initially pull each arm once
        if self.initial_pulls < K:
            self.MAB.pull(self.initial_pulls)
            self.initial_pulls += 1
        else:
            # Calculate the average reward for each arm
            record = self.MAB.get_record()
            successes = record[:, 1]
            total_pulls = record.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                average_rewards = np.true_divide(successes, total_pulls)
                average_rewards[~np.isfinite(average_rewards)] = 0  # Set NaNs and infinities to 0

            # Find the arm(s) with the highest average reward
            max_reward = np.max(average_rewards)
            arms_with_max_reward = np.where(average_rewards == max_reward)[0]

            # Randomly select an arm among those with the maximum average reward
            selected_arm = np.random.choice(arms_with_max_reward)
            # Pull the selected arm
            self.MAB.pull(selected_arm)

class ETC():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.Ne = math.floor(((self.MAB.get_T() * (math.sqrt(math.log(2 * self.MAB.get_K() / self.delta)))) / (2 * self.MAB.get_K()))**(2/3))
        self.exploration_done = np.zeros(self.MAB.get_K())

    def reset(self):
        self.MAB.reset()
        self.exploration_done = np.zeros(self.MAB.get_K())

    def play_one_step(self):
        # Check if the exploration phase is over
        if np.all(self.exploration_done >= self.Ne):
            # Exploitation phase: Choose the arm with the highest empirical mean
            record = self.MAB.get_record()
            successes = record[:, 1]
            total_pulls = record.sum(axis=1)
            average_rewards = successes / total_pulls
            max_reward = np.max(average_rewards)
            arms_with_max_reward = np.where(average_rewards == max_reward)[0]
            selected_arm = np.random.choice(arms_with_max_reward)
        else:
            # Exploration phase: Choose an arm that hasn't been pulled Ne times yet
            arms_less_than_Ne = np.where(self.exploration_done < self.Ne)[0]
            selected_arm = np.random.choice(arms_less_than_Ne)

        # Pull the selected arm and update the exploration count
        self.MAB.pull(selected_arm)
        self.exploration_done[selected_arm] += 1

class Epgreedy():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.initial_pulls = 0  # To track if all arms are pulled once

    def reset(self):
        self.MAB.reset()
        self.initial_pulls = 0

    def play_one_step(self):
        K = self.MAB.get_K()
        t = sum(self.MAB.get_record().sum(axis=1))

        # Initially pull each arm once
        if self.initial_pulls < K:
            self.MAB.pull(self.initial_pulls)
            self.initial_pulls += 1
        else:
            # Calculate epsilon_t
            epsilon_t = (K * math.log(t) / t)**(1/3)

            # Exploration or exploitation
            if np.random.random() < epsilon_t:
                # Exploration: Randomly select an arm
                selected_arm = np.random.choice(K)
            else:
                # Exploitation: Choose the arm with the highest average reward
                record = self.MAB.get_record()
                successes = record[:, 1]
                total_pulls = record.sum(axis=1)
                average_rewards = successes / total_pulls
                max_reward = np.max(average_rewards)
                arms_with_max_reward = np.where(average_rewards == max_reward)[0]
                selected_arm = np.random.choice(arms_with_max_reward)

            # Pull the selected arm
            self.MAB.pull(selected_arm)

class UCB():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        K = self.MAB.get_K()
        t = sum(self.MAB.get_record().sum(axis=1))

        record = self.MAB.get_record()
        successes = record[:, 1]
        total_pulls = record.sum(axis=1)

        # Avoid division by zero
        total_pulls_with_fallback = np.where(total_pulls > 0, total_pulls, 1)

        empirical_means = np.divide(successes, total_pulls_with_fallback)
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.sqrt(np.log(K * t / self.delta) / total_pulls_with_fallback)
            bonus[np.isnan(bonus)] = np.inf  # Assign a high bonus value if the calculation fails

        ucb_values = empirical_means + bonus

        # Select the arm with the highest UCB value, handling the case where all UCB values are NaN or infinite
        if not np.any(np.isfinite(ucb_values)):
            selected_arm = np.random.choice(K)  # Fallback to random selection
        else:
            max_ucb = np.max(ucb_values[np.isfinite(ucb_values)])  # Ignore NaN and infinite values
            selected_arm = np.random.choice(np.where(ucb_values == max_ucb)[0])

        self.MAB.pull(selected_arm)

class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB
        self.alpha = np.ones(self.MAB.get_K())
        self.beta = np.ones(self.MAB.get_K())

    def reset(self):
        self.MAB.reset()
        self.alpha = np.ones(self.MAB.get_K())
        self.beta = np.ones(self.MAB.get_K())

    def play_one_step(self):
        # Sample from the posterior for each arm
        sampled_theta = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]

        # Select the arm with the highest sampled value
        selected_arm = np.argmax(sampled_theta)

        # Pull the selected arm and observe the reward
        reward = self.MAB.pull(selected_arm)

        # Update the posterior parameters for the selected arm
        self.alpha[selected_arm] += reward
        self.beta[selected_arm] += 1 - reward
