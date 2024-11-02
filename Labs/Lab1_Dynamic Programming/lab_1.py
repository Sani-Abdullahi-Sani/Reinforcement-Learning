###
# Group Members
# Sani Abdullahi Sani:  2770930
# Christine Mtaranyika: 2770653
# Liam Culligan:        0715293H
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    while True:
        V = policy_evaluation_fn(env, policy, discount_factor)
        policy_stable = True

        for s in range(env.observation_space.n):
            chosen_a = np.argmax(policy[s])
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[best_a]

        if policy_stable:
            return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    V = np.zeros(env.observation_space.n)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            action_values = one_step_lookahead(s, V)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        best_action = np.argmax(one_step_lookahead(s, V))
        policy[s, best_action] = 1.0

    return policy, V

def plot_running_time(env):
    discount_rates = np.logspace(-0.2, 0, num=30)
    policy_times = []
    value_times = []

    for gamma in discount_rates:
        # Measure the time for policy iteration
        policy_time = timeit.timeit(lambda: policy_iteration(env, discount_factor=gamma), number=10)
        policy_times.append(policy_time / 10)  # Average time over 10 runs

        # Measure the time for value iteration
        value_time = timeit.timeit(lambda: value_iteration(env, discount_factor=gamma), number=10)
        value_times.append(value_time / 10)  # Average time over 10 runs

    
    plt.figure(figsize=(10, 6))
    plt.plot(discount_rates, policy_times, label='Policy Iteration')
    plt.plot(discount_rates, value_times, label='Value Iteration')
    plt.xlabel('Discount Rate (Gamma)')
    plt.ylabel('Average Time (seconds)')
    plt.xscale('log')
    plt.title('Average Running Time for Policy Iteration and Value Iteration')
    plt.legend()
    plt.show()


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    #np.set_printoptions(precision=2, suppress=True)

    # Generate random policy
    random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # Evaluate random policy
    v = policy_evaluation(env, random_policy)

    # Print state value for each state, as grid shape
    print(np.reshape(v, env.shape))

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # Use policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env)

    # Print out best action for each state in grid shape
    action_mapping = ['↑', '→', '↓', '←']
    best_actions = np.argmax(policy, axis=1)
    best_actions_grid = np.array([action_mapping[a] for a in best_actions]).reshape(env.shape)
    print(best_actions_grid)

    # Print state value for each state, as grid shape
    print(np.reshape(v, env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # Use value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)

    # Print out best action for each state in grid shape
    best_actions_vi = np.argmax(policy, axis=1)
    best_actions_grid_vi = np.array([action_mapping[a] for a in best_actions_vi]).reshape(env.shape)
    print(best_actions_grid_vi)

    # Print state value for each state, as grid shape
    print(np.reshape(v, env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    # Plot running times for different discount rates
    plot_running_time(env)

if __name__ == "__main__":
    main()
