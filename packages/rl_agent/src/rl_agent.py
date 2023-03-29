#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from pprint import pprint
import time
import signal

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose, WheelsCmdStamped, Pose2DStamped

def generate_action(max_velocity):
    action = np.random.uniform(0.1, max_velocity, size=2)
    action = np.round(action, 2)
    return action

def generate_action_space(max_velocity, num_actions):
    actions = []
    for i in range(num_actions):
        action = generate_action(max_velocity)
        actions.append(action)
    return actions



class RLAgentNode(DTROS):
    def __init__(self, node_name):
        super(RLAgentNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.namespace = rospy.get_namespace()
        self.rate = rospy.Rate(5)
        self.lane_pose_sub = rospy.Subscriber(str(self.namespace + "lane_filter_node/lane_pose"), LanePose, self.lane_pose_cb)
        self.pose_sub = rospy.Subscriber(str("/donald/velocity_to_pose_node/pose"), Pose2DStamped, self.pose_cb)
        self.rl_agent_pub = rospy.Publisher(str(self.namespace + "wheels_driver_node/wheels_cmd"), WheelsCmdStamped, queue_size=1)
        # register the interrupt signal handler
        signal.signal(signal.SIGINT, self.shutdown)
        self.lane_pose = [0, 0]
        self.pose = [0, 0, 0]
        
    def lane_pose_cb(self, msg):
        actual_dist = np.round(msg.d, 2)
        actual_angle = np.round(msg.phi, 2)
        self.lane_pose = actual_dist, actual_angle

    def pose_cb(self, msg):
        pose_x, pose_y, pose_theta = msg.x, msg.y, msg.theta
        self.pose = [pose_x, pose_y, pose_theta]

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
        self.lr.coef_ = np.random.rand(5, 7)
        # setting the coefficient of x_dist to 0
        self.lr.intercept_ = np.zeros(5)
        self.max_velocity = 0.3
        """self.actions = np.array([
            [self.max_velocity, -self.max_velocity],
            [0, self.max_velocity],
            [self.max_velocity, 0],
            [self.max_velocity, self.max_velocity],
            [0, 0]
        ])"""
        self.actions = generate_action_space(self.max_velocity, 10)
    
    def step(self, action):
        velocities = np.array(action)
        x, y, theta = self.RLAgentNode.pose
        lane_d, lane_phi = self.RLAgentNode.lane_pose
        print("Current state: {}, {}, {}, {}, {}".format(lane_d, lane_phi, x, y, theta))
        print("Current coefs: \n{}".format(self.lr.coef_))
        input_array = np.array([lane_d, lane_phi, x, y, theta, velocities[0], velocities[1]])
        input_array = input_array.reshape(1, -1)
        
        predicted_location = self.lr.predict(input_array)
        predicted_location = predicted_location.reshape(5)
        
        
        
        predicted_lane_d = predicted_location[0]
        predicted_lane_phi = predicted_location[1]
        predicted_x = predicted_location[2]
        predicted_y = predicted_location[3]
        predicted_theta = predicted_location[4]
        reward = 1 - abs(predicted_lane_d) - abs(predicted_lane_phi)
        done = False
        next_state = predicted_location
        print("Action: {}, next state: {}, reward: {}".format(action, next_state, reward))
        return next_state, reward, done, input_array

    
    def reset(self):
        try:
            state = self.RLAgentNode.lane_pose
        except:
            state = [0, 0]
        state = np.round(state, 2)
        return state



class ModelBasedRL:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}

    def exec_action(self, action, action_dict):
        velocities = self.env.actions[action]
        # publishing velocities to robot
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.vel_left = velocities[0]
        wheels_cmd_msg.vel_right = velocities[1]
        print("Publishing: {}".format(velocities))
        self.env.RLAgentNode.rl_agent_pub.publish(wheels_cmd_msg)
        print("published to topic: {}".format(self.env.RLAgentNode.rl_agent_pub.name))

        # get actual data from subscriber
        actual_x, actual_y, actual_theta = self.env.RLAgentNode.pose
        actual_lane_d, actual_lane_phi = self.env.RLAgentNode.lane_pose
        actual_location = np.array([actual_lane_d, actual_lane_phi, actual_x, actual_y, actual_theta])
        input_array = np.array(action_dict[action]).reshape(1, 7)
        actual_location = actual_location.reshape(1, 5)
        pprint(input_array)
        pprint(actual_location)
        self.env.lr.fit(input_array, actual_location)
        
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
                action_dict = {}
                for action_idx in range(len(self.env.actions)):
                    next_state, reward, done , input_array= self.env.step(self.env.actions[action_idx])
                    self.update_Q(state, action_idx, next_state[:1], reward)
                    action_dict[action_idx] = input_array

                action_idx = self.get_action(state)
                self.exec_action(action_idx, action_dict)                
                
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
    rospy.logwarn("RL agent node started. Waiting for lane pose...")
    rospy.wait_for_message(str(env.RLAgentNode.namespace + "lane_filter_node/lane_pose"), LanePose)
    rospy.logwarn("Lane pose received. Starting training...")
    agent.train(1000)
    rospy.spin()
