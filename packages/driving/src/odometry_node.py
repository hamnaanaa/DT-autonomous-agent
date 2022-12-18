#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import WheelEncoderStamped, Pose2DStamped
from duckietown_msgs.srv import SetValue

import message_filters

class OdometryNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """
        super(OdometryNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Initialize the DTROS parent class
        self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
        self._radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 100)
        rospy.loginfo(f"Radius: {self._radius}")
        self._baseline = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline', 100)
        rospy.loginfo(f"Baseline: {self._baseline}")
        self._N_total=135

        self.sign_left = 1
        self.sign_right = 1
        
        # Publishers
        self.pub_pose = rospy.Publisher(f'/{self.veh_name}/odometry_node/pose',Pose2DStamped,queue_size=1)

        # Setup subscribers
        self.sub_encoder_left = message_filters.Subscriber(f'/{self.veh_name}/left_wheel_encoder_node/tick', WheelEncoderStamped)
        self.sub_encoder_right = message_filters.Subscriber(f'/{self.veh_name}/right_wheel_encoder_node/tick', WheelEncoderStamped)

        self.reset_service = rospy.Service(f'/{self.veh_name}/odometry_node/reset', SetValue, self.cb_reset)

        # Setup the time synchronizer
        self.ts_encoders = message_filters.ApproximateTimeSynchronizer(
            [self.sub_encoder_left, self.sub_encoder_right], 10, 0.5)
    
        self.ts_encoders.registerCallback(self.cb_encoder_data)

        self.left_encoder_last = None
        self.right_encoder_last = None
        self.encoders_timestamp_last = None
        self.encoders_timestamp_last_local = None

        self.last_ticks_right = None
        self.last_ticks_left = None
        
        self.x_pos = 0
        self.y_pos = 0

        self.theta = 0

        self.log("Initialized")


    def cb_encoder_data(self, left_encoder, right_encoder):
    #def cb_encoder_data(self, msg, arg):
        """ Update encoder distance information from ticks.
        """

        timestamp_now = rospy.get_time()

        # Use the average of the two encoder times as the timestamp
        left_encoder_timestamp = left_encoder.header.stamp.to_sec()
        right_encoder_timestamp = right_encoder.header.stamp.to_sec()
        timestamp = (left_encoder_timestamp + right_encoder_timestamp) / 2

        if self.last_ticks_left is None:
            self.last_ticks_left = left_encoder.data
            self.last_ticks_right = right_encoder.data
            return

        if not self.left_encoder_last:
            self.left_encoder_last = left_encoder
            self.right_encoder_last = right_encoder
            self.encoders_timestamp_last = timestamp
            self.encoders_timestamp_last_local = timestamp_now
            return

        # Skip this message if the time synchronizer gave us an older message
        dtl = left_encoder.header.stamp - self.left_encoder_last.header.stamp
        dtr = right_encoder.header.stamp - self.right_encoder_last.header.stamp
        if dtl.to_sec() < 0 or dtr.to_sec() < 0:
            self.loginfo("Ignoring stale encoder message")
            return

        dt = timestamp - self.encoders_timestamp_last

        if dt < 1e-6:
            self.logwarn("Time since last encoder message (%f) is too small. Ignoring" % dt)
            return

        delta_ticks_left = (left_encoder.data - self.last_ticks_left)
        delta_ticks_right = (right_encoder.data - self.last_ticks_right)

        self.last_ticks_left = left_encoder.data
        self.last_ticks_right = right_encoder.data

        delta_d_left =  2*np.pi * self._radius * delta_ticks_left  / self._N_total
        delta_d_right = 2*np.pi * self._radius * delta_ticks_right / self._N_total
        
        delta_theta = (delta_d_right - delta_d_left) / self._baseline
        self.theta += delta_theta
        self.theta %= 2*np.pi

        self.x_pos -= 0.5 * (delta_d_left + delta_d_right) * np.sin(self.theta)
        self.y_pos += 0.5 * (delta_d_left + delta_d_right) * np.cos(self.theta)

        # self.loginfo(f"{self.x_pos:.3f}, {self.y_pos:.3f}, {self.theta:.3f}, {delta_ticks_left}, {delta_ticks_right}")

        self.left_encoder_last = left_encoder
        self.right_encoder_last = right_encoder
        self.encoders_timestamp_last = timestamp
        self.encoders_timestamp_last_local = timestamp_now

        pose = Pose2DStamped()
        pose.x = self.x_pos
        pose.y = self.y_pos
        pose.theta = self.theta
        self.pub_pose.publish(pose)

    
    def cb_reset(self, rqst):

        angle = rqst.value
        
        self.left_encoder_last = None
        self.right_encoder_last = None
        self.encoders_timestamp_last = None
        self.encoders_timestamp_last_local = None

        self.x_pos = 0
        self.y_pos = 0

        self.theta = angle

        return []
    
if __name__ == '__main__':
    node = OdometryNode(node_name='odometry_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("wheel_encoder_node is up and running...")
    rospy.spin()
    rate = rospy.Rate(50) # ROS Rate at 5Hz
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()
