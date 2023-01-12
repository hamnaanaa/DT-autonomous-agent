#!/usr/bin/env python3
from enum import Enum

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import FSMState, Twist2DStamped
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern, SetFSMState
from std_msgs.msg import String
from std_srvs.srv import SetBool


# State = pattern name
class State(Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    STOP = "STOP"

    LANE_FOLLOWING = "LANE_FOLLOWING"

    TURN_LEFT = 'TURN_LEFT'
    TURN_RIGHT = 'TURN_RIGHT'
    DRIVE_STRAIGHT = 'DRIVE_STRAIGHT'

    JOYSTICK = "JOYSTICK"
    LANE_JOYSTICK = "LANE_JOYSTICK"


class StateMachineNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(StateMachineNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        self.veh_name = rospy.get_namespace().strip("/")

        self.prefix = f"/{self.veh_name}/state_machine_node/"

        # setup car_cmd_switch (multiplexer for car_cmd)
        mode_topic = self.prefix + "mode"

        SOURCES = {
            State.JOYSTICK: f'/{self.veh_name}/joy_mapper_node/car_cmd',
            State.LANE_FOLLOWING: f'/{self.veh_name}/lane_follow_node/car_cmd',
            State.TURN_LEFT: f'/{self.veh_name}/intersection_node/car_cmd',
            State.TURN_RIGHT: f'/{self.veh_name}/intersection_node/car_cmd',
            State.DRIVE_STRAIGHT: f'/{self.veh_name}/intersection_node/car_cmd',
        }

        # self.led_srv = f"/{self.veh_name}/led_emitter_node/set_pattern"
        # self.led_custom_srv = f"/{self.veh_name}/led_emitter_node/set_custom_pattern"
        # rospy.wait_for_service(self.led_srv)
        # rospy.wait_for_service(self.led_custom_srv)
        # construct publisher --- publish to topic chatter
        self.mode_pub = rospy.Publisher(mode_topic, FSMState, queue_size=10)
        # construct subscriber --- subscribe to topic chatter
        self.sub = rospy.Subscriber(self.prefix + 'mock_state', FSMState, self.cb_change_mode)
        self.mode_service = rospy.Service(self.prefix + "change_mode", SetFSMState, self.cb_change_mode_srv)

        self.wheel_cmd_pub = rospy.Publisher(f"/{self.veh_name}/state_machine_node/cmd", Twist2DStamped, queue_size=1)
        self.cmd_subs = {}
        for source, topic in SOURCES.items():
            self.cmd_subs[source.value] = rospy.Subscriber(topic, Twist2DStamped, self.cb_car_cmd, callback_args=source.value)

        self.current_state = State.LANE_FOLLOWING.value
        # self.set_LEDs(self.current_state)

    def stop_car(self):
        cmd = Twist2DStamped()
        cmd.v = 0.0
        cmd.omega = 0.0
        self.wheel_cmd_pub.publish(cmd)


    def cb_car_cmd(self, cmd, source):
        # forward car_cmd to wheels only if from current state
        if source == self.current_state:
            self.wheel_cmd_pub.publish(cmd)


    def cb_change_mode_srv(self, incoming):
        # wrapper for service with correct return value
        self.cb_change_mode(incoming)
        return []

    def cb_change_mode(self, incoming):
        self.loginfo(f"Request to change state received: {incoming.state}")

        if self.current_state != incoming.state:
            self.stop_car()
            self.current_state = incoming.state
            # self.set_LEDs(self.current_state)

            message = FSMState()
            message.state = self.current_state
            self.loginfo(f"New state after change: {message.state}")
            self.mode_pub.publish(message)

    def set_LEDs(self, state):
        '''
        Predifined patterns:
        `WHITE`, `GREEN`, `BLUE`, `LIGHT_OFF`, `CAR_SIGNAL_PRIORITY`, 'CAR_SIGNAL_SACRIFICE_FOR_PRIORITY'
        `CAR_DRIVING`.
        '''
        # set pattern
        self.loginfo(f"I want to change the LED to {state} of type {type(state)}")
        if state == State.IDLE.value:
            self.set_pattern("LIGHT_OFF")
        elif state == State.JOYSTICK.value or state == State.LANE_FOLLOWING.value:
            self.set_pattern("WHITE")
        elif state == State.WAITING.value:
            self.set_pattern("CAR_SIGNAL_SACRIFICE_FOR_PRIORITY")
        elif state == State.STOP.value:
            self.set_pattern("RED")
        elif state == State.TURN_LEFT.value:
            self.set_pattern("BLUE")
        elif state == State.TURN_RIGHT.value:
            self.set_pattern("BLUE")
        elif state == State.DRIVE_STRAIGHT.value:
            self.set_pattern("BLUE")
        elif state == State.LANE_JOYSTICK.value:
            self.set_pattern("BLUE")

    def set_pattern(self, pattern_name):
        try:
            self.loginfo('Trying to set led pattern to {}'.format(pattern_name))
            if pattern_name.startswith("pattern: "):
                pattern_srv = rospy.ServiceProxy(self.led_custom_srv, SetCustomLEDPattern)
            else:
                pattern_srv = rospy.ServiceProxy(self.led_srv, ChangePattern)
            pattern_srv(String(pattern_name))
            self.loginfo("Finished setting led pattern")
        except rospy.ServiceException as e:
            self.loginfo("Service call failed: %s" % e)

    def on_shutdown(self):
        self.stop_car()
        super(StateMachineNode, self).on_shutdown()


if __name__ == '__main__':
    # create the node
    node = StateMachineNode(node_name='state_machine_node')  # run node
    # node.run()

    # keep spinning
    rospy.spin()
