#!/usr/bin/env python3
import math
from math import sin, cos
import time
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, Pose2DStamped, FSMState
from duckietown_msgs.srv import SetFSMState, SetValue


class IntersectionNode(DTROS):
    def __init__(self, node_name):
        """ Closed-Feedback Waypoint controller using dead-reckoning
            Service expects n  (x,y) waypoints, and a final heading (in odom frame)
        """

        # Initialize the DTROS parent class
        super(IntersectionNode, self).__init__(node_name=node_name,
                                              node_type=NodeType.BEHAVIOR)
        self.veh_name = rospy.get_namespace().strip("/")
        self.prefix = f"/{self.veh_name}/intersection_node/"

        rospy.set_param(self.prefix + "kp", 1.0)
        rospy.set_param(self.prefix + "max_omega", 2.0)

        self.current_pos = (0.,0.)
        
        # pub
        self.wheel_cmd_pub = rospy.Publisher(self.prefix + "car_cmd", Twist2DStamped, queue_size=1)

        # sub
        self.odom_sub = rospy.Subscriber(f"/{self.veh_name}/odometry_node/pose", Pose2DStamped, self.cb_on_odom)
        self.mode_sub = rospy.Subscriber(f"/{self.veh_name}/state_machine_node/mode", FSMState, self.cb_on_mode_change)

        # srv
        change_mode_srv = f"/{self.veh_name}/state_machine_node/change_mode"
        self.mode_srv = rospy.ServiceProxy(change_mode_srv, SetFSMState)

        reset_odom_srv = f"/{self.veh_name}/odometry_node/reset"
        self.reset_odom = rospy.ServiceProxy(reset_odom_srv, SetValue)

        rospy.wait_for_service(reset_odom_srv)
        rospy.wait_for_service(change_mode_srv)

        #self.traj_step = 0
        #self.trajectory = np.array([[0., 0.3], [0.3, 0.3]]) # [2, n]
        #self.final_theta = 1.57 # TODO
        #self.trajectory_done = False

        self.loginfo("Initialized")

    def cb_on_mode_change(self, msg):
        self.loginfo(f"Got state change: {msg}")
        
        if msg.state == "TURN_LEFT":
            self.loginfo("")
            self.turn_left()
        elif msg.state == "TURN_RIGHT":
            self.turn_right()
        elif msg.state == "DRIVE_STRAIGHT":
            self.loginfo("drive straight")
            self.drive_straight()
        else:
            return

        self.mode_srv("LANE_FOLLOWING")

    def turn_left(self):
        self.reset_odom(0.0) # reset Odometry
        rospy.sleep(0.1)

        while self.current_pos[2] < 0.9 * np.pi/2:
            self.drive(0.08, 1.4)
            rospy.sleep(0.05)
        return

    def turn_right(self):
        self.reset_odom(0.0) # reset Odometry
        rospy.sleep(0.1)

        while self.current_pos[2] > - 0.9 * np.pi/2:
            self.drive(0.12, -2.2)
            rospy.sleep(0.05)
        return

    def drive_straight(self):
        self.reset_odom(0.0) # reset Odometry
        rospy.sleep(0.1)

        while self.current_pos[1] < 0.6:
            self.drive(0.1, 0.0)
            rospy.sleep(0.05)
        return

        
    def cb_on_odom(self, odom):
        self.current_pos = (odom.x, odom.y, odom.theta)

    #def cb_on_odom(self, odom):
    #""" executes trajectory of waypoints """
        #current_pose = np.array([[odom.x, odom.y]])
        #rot2D = lambda rot: np.array([[cos(rot), -sin(rot)], [sin(rot), cos(rot)]]).T
        #to_next_wp = np.squeeze((self.trajectory[self.traj_step] - current_pose) @ rot2D(odom.theta))
#
        #self.loginfo(f"({odom.x:.3f}, {odom.y:.3f}, {odom.theta:.3f}) -> ({to_next_wp[0]:.3f},{to_next_wp[1]:.3f})")
#
        #kp = rospy.get_param(self.prefix + "kp")
        ##if self.traj_step != len(self.trajectory):
        #omega = kp * (-to_next_wp[0]) # minimize x of to_next_wp
        #max_omega = rospy.get_param(self.prefix + "max_omega")
        #omega = np.clip(omega, -max_omega, max_omega)
#
        #at_wp = np.linalg.norm(to_next_wp) < 0.05
        #aligned = abs(odom.theta - self.final_theta) < 10/180*np.pi
#
        #v = 0.05 if not at_wp else 0.0
        #self.drive(v, omega)
#
        #if at_wp:
            #self.loginfo("Waypoint reachead")
            #if self.traj_step < len(self.trajectory)-1:
                #self.traj_step += 1
#            
            #else:
                #self.trajectory_done = True
                #self.drive(0.0, 0.0) # FOR TESTING


    def drive(self, v, omega):
        cmd = Twist2DStamped()
        cmd.v = v
        cmd.omega = omega
        self.wheel_cmd_pub.publish(cmd)


    def on_shutdown(self):
        self.drive(0.0, 0.0)
        super(IntersectionNode, self).on_shutdown()


if __name__ == '__main__':
    # Initialize the node
    node = IntersectionNode(node_name='intersection_node')
    rospy.spin()