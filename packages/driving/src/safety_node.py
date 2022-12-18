#!/usr/bin/env python3
from enum import Enum

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_srvs.srv import SetBool
from sensor_msgs.msg import Range


class SafetyNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(SafetyNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        self.veh_name = rospy.get_namespace().strip("/")

        self.prefix = f"/{self.veh_name}/safety_node/"

        self.tof = 1.0

        # deactivate car_cmd_switch_node
        switch_car_cmd = f"/{self.veh_name}/car_cmd_switch_node/switch"
        rospy.wait_for_service(switch_car_cmd)
        switch_car_cmd_switch_node = rospy.ServiceProxy(switch_car_cmd, SetBool)
        switch_car_cmd_switch_node(False)

        # pubs     
        self.wheel_cmd_pub = rospy.Publisher(f"/{self.veh_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        # subs
        self.wheel_cmd_sub = rospy.Subscriber(f"/{self.veh_name}/state_machine_node/cmd", Twist2DStamped, self.cb_car_cmd, queue_size=1)
        self.tof_sub = rospy.Subscriber(f'/{self.veh_name}/front_center_tof_driver_node/range', Range, self.handle_ToF)

        self.loginfo("Initialized")


    def handle_ToF(self, data):
        """ Store the fetched ToF measurement. For out of range measurements, the provided sensor value is >8.0 and the actual value is set to 0.0 instead """
        self.tof = 0.0 if data.range > 1.0 else data.range

    def stop_car(self):
        cmd = Twist2DStamped()
        cmd.v = 0.0
        cmd.omega = 0.0
        self.wheel_cmd_pub.publish(cmd)


    def cb_car_cmd(self, cmd):
        # forward car_cmd to wheels only if from current state
        lower_tof = 0.1
        upper_tof = 0.2
        v = cmd.v
        omega = cmd.omega
        if cmd.v > 0 and  self.tof > upper_tof:
            slow_down_factor = ((self.tof - lower_tof) / (upper_tof-lower_tof))
            slow_down_factor = slow_down_factor if self.tof > 0.0 else 1.0
            v *= slow_down_factor
            omega *= np.clip(slow_down_factor, 0, 100.)
            #self.loginfo(f"sdf: {slow_down_factor}, v_cmd: {cmd.v}, v_safe: {v}")

        modified = Twist2DStamped()        
        modified.omega = omega
        modified.v = v
        self.wheel_cmd_pub.publish(cmd)


if __name__ == '__main__':
    # create the node
    node = SafetyNode(node_name='safety_node')  # run node
    # keep spinning
    rospy.spin()
