#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from pprint import pprint
import time
import signal

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose, Twist2DStamped, Pose2DStamped # type: ignore

class DataCollectorNode(DTROS):
    def __init__(self, node_name):
        super(DataCollectorNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.namespace = rospy.get_namespace()
        self.rate = rospy.Rate(2.5)
        self.lane_pose_sub = rospy.Subscriber(str(self.namespace + "lane_filter_node/lane_pose"), LanePose, self.lane_pose_cb)
        self.pose_sub = rospy.Subscriber(str(self.namespace + "velocity_to_pose_node/pose"), Pose2DStamped, self.pose_cb)
        #self.wheels_cmd_sub = rospy.Subscriber(str(self.namespace + "wheels_driver_node/wheels_cmd_executed"), WheelsCmdStamped, self.wheels_cmd_cb)
        self.rl_agent_pub = rospy.Publisher(str(self.namespace + "joy_mapper_node/car_cmd"), Twist2DStamped, queue_size=1)
        # register the interrupt signal handler
        signal.signal(signal.SIGINT, self.shutdown)
        self.lane_pose = [0, 0]
        self.pose = [0, 0, 0]
        self.vel_cmd = [0, 0]
        self.lane_d_buffer = [0, 0, 0, 0, 0]
        self.lane_phi_buffer = [0, 0, 0, 0, 0]
        self.is_shutdown = False
        self.dt = 0.2

    def lane_pose_cb(self, msg):
        actual_dist = np.round(msg.d, 2)
        actual_angle = np.round(msg.phi, 2)
        self.lane_d_buffer.append(actual_dist)
        self.lane_phi_buffer.append(actual_angle)
        self.lane_d_buffer.pop(0)
        self.lane_phi_buffer.pop(0)
        actual_dist = np.mean(self.lane_d_buffer)
        self.lane_pose = actual_dist, actual_angle

    def pose_cb(self, msg):
        pose_x, pose_y, pose_theta = msg.x, msg.y, msg.theta
        self.pose = [pose_x, pose_y, pose_theta]

    """def wheels_cmd_cb(self, msg):
        self.last_wheels_cmd = [msg.vel_left, msg.vel_right]"""

    def assemble_data(self):
        data = [self.lane_pose[0], self.lane_pose[1], self.vel_cmd[0], self.vel_cmd[1]]
        return data
    
    def update_dt(self, dt):
        self.dt = dt

    def shutdown(self, signal, frame):
        # wheels_cmd_msg = WheelsCmdStamped(vel_left=0, vel_right=0)
        twist_msg = Twist2DStamped(v=0, omega=0)
        self.rl_agent_pub.publish(twist_msg)
        rospy.logerr("[DataCollector] Shutdown complete.")
        time.sleep(1)
        self.is_shutdown = True
    
if __name__ == '__main__':
    data_collector_node = DataCollectorNode(node_name='data_collector_node')
    rospy.wait_for_message(str(data_collector_node.namespace + "lane_filter_node/lane_pose"), LanePose)
    with open("data.txt", "w") as f:
        f.write("")
    rospy.loginfo("[Collector] Ready to collect data.")
    previous_state = data_collector_node.assemble_data()
    # while singal is not interrupted
    while not data_collector_node.is_shutdown:
        # publish random velocities
        time_before = time.time()
        new_v = np.random.uniform(0.15, 0.4)
        new_omega = np.random.uniform(-3, 3)
        velocities = np.array([new_v, new_omega])
        velocities = np.round(velocities, 2)
        twist_msg = Twist2DStamped()
        twist_msg.v = velocities[0]
        twist_msg.omega = velocities[1]
        twist_msg.header.stamp = rospy.get_rostime()
        data_collector_node.vel_cmd = velocities
        data_collector_node.rl_agent_pub.publish(twist_msg)
        print("Published: ", velocities)
        data_collector_node.rate.sleep()
        time_after = time.time()
        data_collector_node.update_dt(time_after - time_before)
        # collect data
        current_state = data_collector_node.assemble_data()[:2]
        previous_state.append(data_collector_node.dt)
        print("previous state: ", previous_state)
        print("current state: ", current_state)
        # save data to file
        print("Saving data to file...")
        with open("data.txt", "a") as f:
            f.write(str(previous_state) + "/" + str(current_state) + "\n")
        print("Data saved.") 
        previous_state = data_collector_node.assemble_data()