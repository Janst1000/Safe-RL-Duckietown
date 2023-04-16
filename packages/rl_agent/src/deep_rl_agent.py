import numpy as np
import random
import tensorflow as tf

class DQLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_rate=0.95, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()

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
            print("Random action")
            return random.randrange(self.action_size)
        #print("State shape act: ", state.shape)
        print("Learned Action")
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        print("Replaying")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_values = self.model.predict(state, verbose=0)
            q_values[0][action] = reward if done else reward + self.discount_rate * np.max(self.model.predict(next_state)[0])
            self.model.fit(state, q_values, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("Epsilon: ", self.epsilon)
