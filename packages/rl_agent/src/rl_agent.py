import numpy as np
import rospy
from sklearn.linear_model import LinearRegression
from pprint import pprint

class RobotEnv:
    def __init__(self):
        self.current_dist = 0
        self.current_angle = 0
        self.lr = LinearRegression()
        self.lr.coef_ = np.array([[1, 0.5], [0.5, 1]])
        self.lr.intercept_ = np.array([0, 0])
        self.max_velocity = 0.2
        self.actions = np.array([
            [-self.max_velocity, -self.max_velocity],
            [-self.max_velocity, self.max_velocity],
            [self.max_velocity, -self.max_velocity],
            [self.max_velocity, self.max_velocity]
        ])
    
    def step(self, action):
        velocities = np.array(action)
        distances = self.lr.predict(velocities.reshape(1, -1))[0]
        next_dist = distances[0]
        next_angle = distances[1]
        reward = 1 - abs(next_dist) - abs(next_angle)
        done = False
        self.current_dist = next_dist
        self.current_angle = next_angle
        next_state = np.array([next_dist, next_angle])
        print("Action: {}, next state: {}, reward: {}".format(action, next_state, reward))
        return next_state, reward, done, {}
    
    def reset(self):
        self.current_dist = np.random.uniform(-0.5, 0.5)
        self.current_angle = np.random.uniform(-0.1, 0.1)
        state = np.array([self.current_dist, self.current_angle])
        state = np.round(state, 2)
        return state

class ModelBasedRL:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}

        
        
    def get_action(self, state):
        print("Current Q-table:")
        pprint(self.Q)
        if np.random.uniform(0, 1) < self.epsilon or self.Q == {}:
            print("Exploration")
            # Exploration: choose a random action
            action_idx = np.random.choice(len(self.env.actions))
            print("action_idx: {}".format(action_idx))
            action = self.env.actions[action_idx]
        else:
            print("Exploitation")
            # Exploitation: choose the action with highest Q-value
            state_idx = self.discretize_state(state)
            if state_idx not in self.Q:
                self.Q[state_idx] = np.zeros((2, 2))
            q_values = self.Q[state_idx]
            max_q = np.max(q_values)
            max_indices = np.where(q_values == max_q)[0]
            action_idx = np.random.choice(max_indices)
            action = self.env.actions[action_idx]
        
        return action_idx

    def discretize_state(self, state):
        # Discretize state to indices of a Q-table
        return tuple(np.round(state, 2))

    def update_Q(self, state, action_idx, next_state, reward):
        state_tuple = self.discretize_state(state)
        next_state_tuple = self.discretize_state(next_state)
        if state_tuple not in self.Q:
            self.Q[state_tuple] = np.zeros((4,))
        if next_state_tuple not in self.Q:
            self.Q[next_state_tuple] = np.zeros((4,))
        max_Q = np.max(self.Q[next_state_tuple])
        
        self.Q[state_tuple][action_idx] += self.alpha * (reward + self.gamma * max_Q - self.Q[state_tuple][action_idx])


    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            rospy.loginfo("Episode: {}".format(episode))
            state = self.env.reset()
            done = False
            while not done:
                print("Current state: {}".format(state))
                action_idx = self.get_action(state)
                next_state, reward, done, _ = self.env.step(env.actions[action_idx])
                self.update_Q(state, action_idx, next_state, reward)
                state = next_state

if __name__ == '__main__':
    env = RobotEnv()
    agent = ModelBasedRL(env)
    agent.train(1000)
