#!/usr/bin/env python3
import math
import time
import numpy as np
import rospy



from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped
from custom_msgs.msg import LaneDetection


class LaneFollowNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneFollowNode, self).__init__(node_name=node_name,
                                              node_type=NodeType.BEHAVIOR)
        self.veh_name = rospy.get_namespace().strip("/")
        
        # params
        rospy.set_param(f"/{self.veh_name}/lane_follow_node/kp", 0.6)
        rospy.set_param(f"/{self.veh_name}/lane_follow_node/v", 0.1)
        rospy.set_param(f"/{self.veh_name}/lane_follow_node/max_omega", 3.0)

        # vars
        self.t_last_cmd = rospy.get_time()

        # just publish directly to wheel node
        self.wheel_cmd_pub = rospy.Publisher(f"/{self.veh_name}/lane_follow_node/car_cmd", Twist2DStamped, queue_size=1)
        self.lane_det_sub = rospy.Subscriber(f"/{self.veh_name}/lane_perception_node/lane_det", LaneDetection, self.cb_line_det)

        self.loginfo("Initialized")

    def t_since_last_cmd(self):
        return rospy.get_time() - self.t_last_cmd

    def cb_line_det(self, msg):
        #self.log(f"Got {msg}")
        #white = msg.white_angle
        #yellow = msg.yellow_angle
        horizontal = msg.mid_point_h - 0.5
        vertical = 1.0 - msg.mid_point_v

        if horizontal == 0.0:
            return

        vec = horizontal / vertical

        # if straight white==180-yellow, otherwise correct
        e = - vec

        self._kp = rospy.get_param(f"/{self.veh_name}/lane_follow_node/kp")

        omega = self._kp * e
        
        max_omega = rospy.get_param(f"/{self.veh_name}/lane_follow_node/max_omega")
        omega = np.clip(omega, -max_omega, max_omega)

        #self.loginfo(f"vec: {vec}, omega: {omega}")
        
        #self.loginfo(f"White: {white*180/np.pi:.1f}, yellow: {yellow*180/np.pi:.1f}, e: {e:.1f}")
        self.drive(rospy.get_param(f"/{self.veh_name}/lane_follow_node/v"), omega)


    def drive(self, v, omega):
        cmd = Twist2DStamped()
        cmd.v = v
        cmd.omega = omega
        self.wheel_cmd_pub.publish(cmd)
        self.t_last_cmd = rospy.get_time()

    def on_shutdown(self):
        self.drive(0.0, 0.0)
        super(LaneFollowNode, self).on_shutdown()


if __name__ == '__main__':
    # Initialize the node
    node = LaneFollowNode(node_name='lane_follow_node')
    

    rate = rospy.Rate(12) # ROS Rate at 5Hz
    
    while not rospy.is_shutdown():
        if node.t_since_last_cmd() > 1.0:
            # node.loginfo("CMD timeout")
            node.drive(0.0, 0.0)

        rate.sleep()