#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String


class StateSubscriber(DTROS):
    """
    A dummy node that demonstrates how to subscribe to messages from a state machine's topic
    Notice: it's not necessary to have a run integration of this node in the shell launcher
    """

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(StateSubscriber, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        # construct subscriber --- subscribe to topic chatter
        self.sub = rospy.Subscriber('mock_state_change', String, self.callback)

    def callback(self, data):
        rospy.loginfo(
            "[state_subscriber_mock] New state in state subscriber %s", data.data)


if __name__ == '__main__':
    # create the node
    node = StateSubscriber(node_name='state_subscriber_mock')  # run node
    # keep spinning
    rospy.spin()
