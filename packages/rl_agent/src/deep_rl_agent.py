import numpy as np
import random
import tensorflow as tf
from SafetyLayer import SafetyLayer
from sklearn.linear_model import LinearRegression


def reward_function(lane_d, lane_phi):
    reward = 1 - abs((lane_d**2) * 3) - abs((lane_phi**2)* 1)
    return reward
class DQLAgent:
    def __init__(self, state_size, action_size, actions, learning_rate=0.001, discount_rate=0.95, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01, sleep_time=0.4, safety_enabled=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()
        self.safety_layer = SafetyLayer(max_lane_d=0.07, max_lane_phi=0.8, max_tof_d=100)
        self.lr = LinearRegression()
        self.sleep_time = sleep_time
        self.safety_rate = 1.0
        self.unsafe_actions = 0

        # loading model from file
        root_path = "/code/catkin_ws/src/Safe-RL-Duckietown/packages/rl_agent/config/"
        model_arrays = np.load(str(root_path + "model.npz"))
        self.lr.coef_ = model_arrays["coef"]
        self.lr.intercept_ = model_arrays["intercept"]

        self.safety_enabled = safety_enabled
        self.actions = actions


    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        #print("model built")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if self.safety_enabled:
                safe = False
                action_list = self.actions.copy()
                while not safe:
                    action = random.randrange(len(action_list))
                    predicted_state = self.predict_state_lr(state, self.actions[action])
                    safe = self.safety_layer.check_safety(predicted_state)
                    action_list = action_list.remove(action_list[action])
                    if safe:
                        print("Random action")
                        return action
                    if action_list == [] or action_list is None:
                        return self.get_back_to_safety(state)
                    print("Unsafe action, trying again")
            else:
                print("Random action")
                return random.randrange(self.action_size)
            
        q_values = self.model.predict(state, verbose=0)
        if self.safety_enabled:
            action_list = self.actions.copy()
            # sorting actions by q_values from high to low
            action_list = [x for _, x in sorted(zip(q_values[0], action_list), reverse=True)]
            for action in action_list:
                predicted_state = self.predict_state_lr(state, action)
                safe = self.safety_layer.check_safety(predicted_state)
                if safe:
                    print("Learned Action")
                    # get index of action on self.actions
                    return self.actions.index(action)
                else:
                    print("Unsafe action, trying again")
                    self.iter = 0
                    action = self.optimize(state, action)
                    if action is not None:
                        return action
            # if no safe action was found we slowly drive backwards
            original_action = np.argmax(q_values[0])
            predicted_state = self.predict_state_lr(state, self.actions[original_action]).reshape(2,)
            return self.get_back_to_safety(state, predicted_state, original_action)
        else:
            print("Learned Action")
            return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        print("Replaying")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_values = self.model.predict(state, verbose=0)
            q_values[0][action] = reward if done else reward + self.discount_rate * np.max(self.model.predict(next_state)[0])
            # decreasing reward if action resulted in unsafe state
            if self.safety_enabled:
                safe = self.safety_layer.check_safety(next_state)
                if not safe:
                    q_values[0][action] = -1
                    self.unsafe_actions += 1
                    # updating safety rate
                    self.safety_rate = 1 - (self.unsafe_actions / len(self.memory))
            self.model.fit(state, q_values, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("Epsilon: ", self.epsilon)

    def predict_state_lr(self, state, action):
        state = state.reshape(2,)
        input_array = np.append(state, action)
        input_array = np.append(input_array, self.sleep_time).reshape(1, -1)
        new_state = self.lr.predict(input_array)
        return new_state
    
    def optimize(self, state, action):
        if self.iter > 3:
            return None
        # generation a new random action that is in the range of the old action
        new_v = action[0] / 2 
        new_omega = action[1] / 2
        new_action = [new_v, new_omega]
        # predicting new state
        new_state = self.predict_state_lr(state, new_action)
        print("Predicted state: ", new_state, " with action: ", new_action, "")
        # checking if new state is safe
        safe = self.safety_layer.check_safety(new_state)
        if safe:
            return new_action
        else:
            print("New action {} is unsafe, trying again".format(new_action))
            self.iter += 1
            return self.optimize(state, new_action)
    
    def get_back_to_safety(self, state, original_pred_state=None, original_action=None):
        if type(original_pred_state) == list and type(original_action) == list: 
            #original_pred_state = original_pred_state.reshape(2,)
            original_reward = reward_function(original_pred_state[0], original_pred_state[1])
            new_action = self.go_backwards(state)
            predicted_state = self.predict_state_lr(state, new_action)
            new_reward = reward_function(predicted_state[0], predicted_state[1])
            if new_reward > original_reward:
                print("New action {} is better than original action".format(new_action))
                return new_action
            else:
                print("New action {} is worse than original action".format(new_action))
                return original_action
        else:
            new_action = self.go_backwards(state)
        
        return new_action
    
    def go_backwards(self, state):
        state = state.reshape(2,)
        if state[0] < 0 and state[1] < 0 or state[0] > 0 and state[1] < 0:
            new_action = [-0.15, 0]
        elif state[0] < 0 and state[1] > 0 or state[0] > 0 and state[1] > 0:
            new_action = [0.15, 0] 
        else:
            new_action = [0.15, 0]
        return new_action
