#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String


class StateChanger(DTROS):
    """
    A dummy node that demonstrates how to publish a message to a state machine's topic
    Notice: it's not necessary to have a run integration of this node in the shell launcher
    """

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(StateChanger, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher --- publish to topic mock_state
        self.pub = rospy.Publisher('mock_state', String, queue_size=10)

    def run(self):
        # publish message every 5 seconds
        rate = rospy.Rate(0.1)  # 3Hz
        counter = 0
        while not rospy.is_shutdown():
            states = ["IDLE", "MANUAL_CONTROL", "LANE_FOLLOWING",
                      "STOP", "INTERSECTION_TRAVERSAL", "COORDINATION"]
            while counter < len(states):
                message = states[counter]

                rospy.loginfo(
                    "[state_changer_mock] Change state to: '%s'" % message)
                self.pub.publish(message)
                counter += 1
                rospy.loginfo("[state_changer_mock] Counter: '%s'" % counter)
                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = StateChanger(node_name='state_changer_mock')  # run node
    node.run()

    # keep spinning
    rospy.spin()
