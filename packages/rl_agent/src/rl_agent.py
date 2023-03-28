#!/usr/bin/env python3
import numpy as np
import rospy
from sklearn.linear_model import LinearRegression
from pprint import pprint
import time
import signal

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose, WheelsCmdStamped

class RLAgentNode(DTROS):
    def __init__(self, node_name):
        super(RLAgentNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.namespace = rospy.get_namespace()
        self.rate = rospy.Rate(10)
        self.lane_pose_sub = rospy.Subscriber(str(self.namespace + "lane_filter_node/lane_pose"), LanePose, self.lane_pose_cb)
        self.rl_agent_pub = rospy.Publisher(str(self.namespace + "wheels_driver_node/wheels_cmd"), WheelsCmdStamped, queue_size=1)
        # register the interrupt signal handler
        signal.signal(signal.SIGINT, self.shutdown)
        self.lane_pose = [0, 0]
        
    def lane_pose_cb(self, msg):
        actual_dist = np.round(msg.d * 20, 2) / 20
        actual_angle = np.round(msg.phi * 20, 2) / 20
        self.lane_pose = actual_dist, actual_angle

    def shutdown(self, signal, frame):
        wheels_cmd_msg = WheelsCmdStamped(vel_left=0, vel_right=0)
        self.rl_agent_pub.publish(wheels_cmd_msg)
        rospy.logerr("[RLAgentNode] Shutdown complete.")
        time.sleep(1)
        

        

class RobotEnv:
    def __init__(self):
        self.RLAgentNode = RLAgentNode("RLAgentNode")
        self.current_dist = 0
        self.current_angle = 0
        self.lr = LinearRegression()
        self.lr.coef_ = np.random.rand(2, 2)
        self.lr.intercept_ = np.array([0, 0])
        self.max_velocity = 0.2
        self.actions = np.array([
            [self.max_velocity, -self.max_velocity],
            [0, self.max_velocity],
            [self.max_velocity, 0],
            [self.max_velocity, self.max_velocity],
            [0, 0]
        ])
    
    def step(self, action):
        velocities = np.array(action)
        distances = self.lr.predict(velocities.reshape(1, -1))[0]
        next_dist = distances[0]
        next_angle = distances[1]
        
        next_dist, next_angle = self.lr.predict(velocities.reshape(1, -1))[0]
        # publishing velocities to robot
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.vel_left = velocities[0]
        wheels_cmd_msg.vel_right = velocities[1]
        print("Publishing: {}".format(velocities))
        self.RLAgentNode.rl_agent_pub.publish(wheels_cmd_msg)

        # get actual data from subscriber
        current_pose = self.RLAgentNode.lane_pose
        actual_dist = current_pose[0]
        actual_angle = current_pose[1]
        self.lr.fit([[self.current_dist, self.current_angle], velocities], [[actual_dist, actual_angle], velocities])

        reward = 1 - abs(actual_dist*10) - abs(actual_angle)
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
    
    """def lane_pose_cb(self):
        self.actual_dist = self.RLAgentNode.actual_dist
        self.actual_angle = self.RLAgentNode.actual_angle"""


class ModelBasedRL:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}

        
        
    def get_action(self, state):
        # print("Current Q-table:")
        # pprint(self.Q)
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
                self.Q[state_idx] = np.zeros((len(self.env.actions),))
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
            self.Q[state_tuple] = np.zeros((len(self.env.actions),))
        if next_state_tuple not in self.Q:
            self.Q[next_state_tuple] = np.zeros((len(self.env.actions),))
        max_Q = np.max(self.Q[next_state_tuple])
        self.Q[state_tuple][action_idx] += self.alpha * (reward + self.gamma * max_Q - self.Q[state_tuple][action_idx])


    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            rospy.loginfo("Episode: {}".format(episode))
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                current_time = rospy.Time.now()
                print("Current state: {}".format(state))
                action_idx = self.get_action(state)
                next_state, reward, done, _ = self.env.step(self.env.actions[action_idx])
                self.update_Q(state, action_idx, next_state, reward)
                total_reward += reward
                state = self.env.RLAgentNode.lane_pose
                elaped_ms = (rospy.Time.now() - current_time).to_nsec() / 1000000
                self.env.RLAgentNode.rate.sleep()
                print("Time elapsed: {}ms and Total Reward: {}".format(elaped_ms, total_reward))
                print("Q size: {}".format(len(self.Q)))
                if total_reward >= 100 or total_reward < 0:
                    done = True

                

if __name__ == '__main__':
    time.sleep(3)
    env = RobotEnv()
    agent = ModelBasedRL(env)
    rospy.logwarn("RL agent node started")
    rospy.wait_for_message(str(env.RLAgentNode.namespace + "lane_filter_node/lane_pose"), LanePose)
    agent.train(1000)
    rospy.spin()
