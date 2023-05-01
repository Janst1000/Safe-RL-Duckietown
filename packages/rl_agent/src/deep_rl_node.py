#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from pprint import pprint
import time
import signal

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose, WheelsCmdStamped, Pose2DStamped, Twist2DStamped # type: ignore
from duckietown_msgs.srv import ChangePattern # type: ignore
from std_msgs.msg import String
from rl_agent.msg import SafetyMsg
import sys

# append this directory to path
sys.path.append("/code/catkin_ws/src/Safe-RL-Duckietown/packages/rl_agent/src")
from deep_rl_agent import DQLAgent
from SafetyLayer import SafetyLayer

RATE = 5
SLEEP_TIME = 1.0 / RATE

# Some helper functions
def generate_action(max_v, max_omega):
    action_v = np.random.uniform(0.2, max_v)
    action_omega = np.random.uniform(-max_omega, max_omega)
    action = [action_v, action_omega]
    action = np.round(action, 2)
    return action.tolist()

def generate_action_space(max_v, max_omega, num_actions):
    actions = []
    if num_actions > 3:
        num_actions -= 3
        for i in range(num_actions):
            action = generate_action(max_v, max_omega)
            actions.append(action)

    actions.append([0.3, 0])
    actions.append([0, 2.5])
    actions.append([0, -2.5])
    return actions

def reward_function(lane_d, lane_phi):
    reward = 1 - abs((lane_d**2) * 3) - abs((lane_phi**2)* 1)
    return reward


class DeepRLNode(DTROS):
    def __init__(self, node_name):
        super(DeepRLNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.namespace = rospy.get_namespace()
        self.rate = rospy.Rate(RATE)
        self.lane_pose_sub = rospy.Subscriber(str(self.namespace + "lane_filter_node/lane_pose"), LanePose, self.lane_pose_cb)
        self.pose_sub = rospy.Subscriber(str(self.namespace + "velocity_to_pose_node/pose"), Pose2DStamped, self.pose_cb)
        self.rl_agent_pub = rospy.Publisher(str(self.namespace + "joy_mapper_node/car_cmd"), Twist2DStamped, queue_size=1)
        self.safety_rate_pub = rospy.Publisher(str(self.namespace + "rl_agent_node/safety_rate"), SafetyMsg, queue_size=1)
        self.safety_enabled = rospy.get_param(str(self.namespace + "rl_agent_node/safety_enabled"))
        # register the interrupt signal handler
        signal.signal(signal.SIGINT, self.shutdown)
        self.lane_pose = [0, 0]
        self.pose = [0, 0, 0]
        self.lane_d_buffer = [0, 0, 0, 0, 0]
        self.lane_phi_buffer = [0, 0, 0, 0, 0]
        self.adjusted_lane_pose_pub = rospy.Publisher(str(self.namespace + "rl_agent_node/adjusted_lane_pose"), LanePose, queue_size=1)

        # Turning the LEDS to WHITE on startup
        self.led_service = rospy.ServiceProxy(str(self.namespace) + 'led_emitter_node/set_pattern', ChangePattern)
        # Define the pattern name
        pattern_name = String()
        pattern_name.data = "WHITE"

        # Call the service with the request message
        try:
            response = self.led_service(pattern_name)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            rospy.logwarn("Continuing without LED control")

        
    def lane_pose_cb(self, msg):
        actual_dist = np.round(msg.d, 2)
        actual_angle = np.round(msg.phi, 2)
        try:
            self.lane_d_buffer.append(actual_dist)
            self.lane_phi_buffer.append(actual_angle)
            self.lane_d_buffer.pop(0)
            self.lane_phi_buffer.pop(0)
            actual_dist = np.mean(self.lane_d_buffer)
            actual_angle = np.mean(self.lane_phi_buffer)
        except:
            pass
        self.lane_pose = np.array([actual_dist, actual_angle])
        adjusted_lane_pose = LanePose()
        adjusted_lane_pose.header = msg.header
        adjusted_lane_pose.d = actual_dist
        adjusted_lane_pose.phi = actual_angle
        adjusted_lane_pose.in_lane = msg.in_lane
        try:
            self.adjusted_lane_pose_pub.publish(adjusted_lane_pose)
        except:
            pass

    def pose_cb(self, msg):
        pose_x, pose_y, pose_theta = msg.x, msg.y, msg.theta
        self.pose = [pose_x, pose_y, pose_theta]

    def shutdown(self, signal, frame):
        # wheels_cmd_msg = WheelsCmdStamped(vel_left=0, vel_right=0)
        twist_msg = Twist2DStamped(v=0, omega=0)
        self.rl_agent_pub.publish(twist_msg)
        pattern_name = String()
        pattern_name.data = "LIGHT_OFF"
        response = self.led_service(pattern_name)
        rospy.logerr("[DeepRLNode] Shutdown complete.")
        time.sleep(1)

class RobotEnv:
    def __init__(self):
        self.DeepRLNode = DeepRLNode("DeepRLNode")

        self.max_v = 0.3
        self.max_omega = 2.5
        #self.actions = [[0.4, 0], [0.2, 4] ,[0.2, -4], [0, 3], [0, -3]]
        
        self.actions = generate_action_space(self.max_v, self.max_omega, 100)
        
    def get_state(self):
        state = self.DeepRLNode.lane_pose
        state = np.append(state, SLEEP_TIME)
        state = np.array(state).reshape(1, 3)
        return state
    
    def exec_action(self, action):
        if type(action) == list:
            velocities = action
        elif type(action) == np.array:
            action.reshape(2,)
            velocities = action.tolist()
        else:
            velocities = self.actions[action]
        twist_msg = Twist2DStamped()
        twist_msg.v = velocities[0]
        twist_msg.omega = velocities[1]
        twist_msg.header.stamp = rospy.Time.now()
        #print("Publishing: {}".format(velocities))
        self.DeepRLNode.rl_agent_pub.publish(twist_msg)
        # get actual data from subscriber
        actual_lane_d, actual_lane_phi = self.DeepRLNode.lane_pose
        reward = reward_function(actual_lane_d, actual_lane_phi)
        return velocities, reward
    
    def safety_stop(self):
        # stopping the robot if all actions are not safe
        twist_msg = Twist2DStamped()
        twist_msg.v = 0
        twist_msg.omega = 0
        twist_msg.header.stamp = rospy.Time.now()
        self.DeepRLNode.rl_agent_pub.publish(twist_msg)
    
    def predict_state_cs(self, state, action):
        state = state.reshape(2,)
        d = state[0]
        phi = state[1]
        v = self.actions[action][0]
        omega = self.actions[action][1]
        dt = SLEEP_TIME
        new_d = d + v * np.sin(phi) * dt
        new_phi = phi + omega * dt
        new_state = np.array([new_d, new_phi]).reshape(1, 2)
        return new_state
    
    def reset(self):
        try:
            state = self.DeepRLNode.lane_pose
        except:
            state = [0, 0]
        state = np.round(state, 2)
        return state

if __name__ == '__main__':
    time.sleep(3)
    env = RobotEnv()
    agent = DQLAgent(state_size=2, action_size=len(env.actions), actions = env.actions, sleep_time=SLEEP_TIME, safety_enabled=env.DeepRLNode.safety_enabled)
    if env.DeepRLNode.safety_enabled:
        rospy.logwarn("Safety Layer is enabled.")
    else:
        rospy.logwarn("Safety Layer is disabled.")
    rospy.logwarn("RL agent node started. Waiting for lane pose...")
    rospy.wait_for_message(str(env.DeepRLNode.namespace + "lane_filter_node/lane_pose"), LanePose)
    rospy.logwarn("Lane pose received. Starting training...")
    num_episode = 1000
    batch_size = int(RATE * 10) # execute for 5 seconds before we train the model
    total_reward = 0


    for episode in range(num_episode):
        total_reward = 0
        done = False
        steps = 0
        rospy.logdebug("Episode: {}".format(episode))
        while not done:
            #print("Episode: {}".format(episode))
            current_state = env.get_state()
            #print("Current state: {}".format(current_state))
            action = agent.act(current_state)
            if action is None:
                rospy.logerr("No safe action found.")
                env.safety_stop()
                time.sleep(SLEEP_TIME)
                continue
            velocities, reward = env.exec_action(action)
            total_reward += reward
            print("State: {}, Action: [{}, {}], Reward: {}, Total Reward: {}".format(current_state, velocities[0], velocities[1], reward, total_reward))
            if total_reward >= 50:
                done=True
            time.sleep(SLEEP_TIME)
            if velocities in env.actions:
                # converting action to index in case it is in the action list
                if type(action) == list:
                    action = env.actions.index(action)
                agent.remember(current_state, action, reward, env.get_state(), done)
                steps += 1
            env.safety_stop()
            if steps >= batch_size:
                safety_rate = agent.replay(batch_size)
                safety_msg = SafetyMsg()
                safety_msg.safety_rate = safety_rate
                safety_msg.epsilon = agent.epsilon
                safety_msg.header.stamp = rospy.Time.now()
                env.DeepRLNode.safety_rate_pub.publish(safety_msg)
                steps = 0
            if done == True:
                safety_rate = agent.replay(steps)
                safety_msg = SafetyMsg()
                safety_msg.safety_rate = safety_rate
                safety_msg.epsilon = agent.epsilon
                safety_msg.header.stamp = rospy.Time.now()
                env.DeepRLNode.safety_rate_pub.publish(safety_msg)
                steps = 0
        batch_size += 2
            
        
    rospy.spin()