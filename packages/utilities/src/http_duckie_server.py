#!/usr/bin/env python3
import os
import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from duckietown_msgs.msg import FSMState, Pose2DStamped
from duckietown_msgs.srv import SetFSMState
from sensor_msgs.msg import Range

from flask import Flask, request, jsonify
import random
import json

app = Flask(__name__)
node = None

mock_state = {
    "position": {
        "x": 0,
        "y": 0,
        "theta": 0
    },
    "status": {
        "battery": 50.0,
        "speed": 5.0,
        "state": "LANE_FOLLOWING",
        "lights": {
            "frontLeft": {
                "isBlinking": False,
                "color": "0xFFFFFF"
            },
            "frontRight": {
                "isBlinking": False,
                "color": "0xFFFFFF"
            },
            "backLeft": {
                "isBlinking": False,
                "color": "0xFF2E00"
            },
            "backRight": {
                "isBlinking": False,
                "color": "0xFF2E00"
            }
        },
        "distanceSensor": 10.0
    }
}


@app.route('/status', methods=['GET'])
def index():
    if node is None:
        json_status = json.dumps(mock_state)
        return json_status
    else:

        json_status = json.dumps(node.get_status())
        return json_status


@app.route('/route', methods=['POST'])
def handle_POST():
    if node is None:
        rospy.loginfo("Node is not initialized")
        return {'status': 'node not initialized'}
    # rospy.loginfo(f"Got a POST request with JSON ROUTE: {request.json}")
    route = request.json['route']
    # rospy.loginfo(f"Received route: {route}")
    node.set_route(route)
    node.update_state('LANE_FOLLOWING')
    # rospy.loginfo(f"Done updating state of the state machine")
    return {'status': 'route received'}


@app.route('/state', methods=['PUT'])
def handle_PUT():
    if node is None:
        rospy.loginfo("Node is not initialized")
        return {'state': 'node not initialized'}
    rospy.loginfo(
            f"[HTTP Server] Got a PUT request with JSON: {request.json}")
    new_state = request.json['state']
    rospy.loginfo(
        f"[HTTP Server] Parsed payload will be given to state machine: {new_state}")
    if new_state == "CONTINUE":
        if node.route[0] == "IL":
            node.update_state('TURN_LEFT')
        elif node.route[0] == "IR":
            node.update_state('TURN_RIGHT')
        elif node.route[0] == "IS":
            node.update_state('DRIVE_STRAIGHT')
        if len(node.route) > 0:
            node.route.pop(0)
        else:
            rospy.loginfo("Route is empty")

    else:
        node.update_state(new_state)
    rospy.loginfo(f"[HTTP Server] Done updating state of the state machine")
    return {'state': new_state}


class HTTPDuckieServer(DTROS):

    def __init__(self, node_name):
        # --- DTROS init ---
        super(HTTPDuckieServer, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")

        # --- Server setup ---
        self.server = app
        
        # --- Route init ---
        self.route = []

        # --- Sensor reading ---
        # Position
        self.sub_pose = rospy.Subscriber(
            f'/{self.veh_name}/odometry_node/pose', Pose2DStamped, self.handle_pose)
        self.pose = {
            "x": 0,
            "y": 0,
            "theta": 0
        }
        # ToF
        self.tof = 0.0
        self.sub_tof = rospy.Subscriber(
            f'/{self.veh_name}/front_center_tof_driver_node/range', Range, self.handle_ToF)
        # Battery (mocked)
        self.battery_level = 100.0
        # Speed TODO: fetch when implemented from odometry
        self.speed = 0.0
        # State
        self.sub_fsm = rospy.Subscriber(
            f'/{self.veh_name}/state_machine_node/mode', FSMState, self.handle_FSM)
        self.state = "IDLE"
        # Lights TODO: fetch from the lights
        self.lights = {
            "frontLeft": {
                "isBlinking": False,
                "color": "0xFFFFFF"
            },
            "frontRight": {
                "isBlinking": False,
                "color": "0xFFFFFF"
            },
            "backLeft": {
                "isBlinking": False,
                "color": "0xFF2E00"
            },
            "backRight": {
                "isBlinking": False,
                "color": "0xFF2E00"
            }
        }

        self.get_status()

    def start_server(self):
        app.run(host='0.0.0.0', port=1318)

    def get_status(self):
        # Update the internal representation with the actual values
        self.agent_status = {
            "position": self.pose,
            "status": {
                "battery": self.battery_level,
                "speed": self.speed,
                "state": self.state,
                "lights": self.lights,
                "distanceSensor": self.tof
            }
        }
        return self.agent_status
    
    def set_route(self, route):
        self.route = route

    def update_state(self, new_state):
        rospy.wait_for_service(
            f"/{self.veh_name}/state_machine_node/change_mode")
        state_machine_switch_node = rospy.ServiceProxy(
            f"/{self.veh_name}/state_machine_node/change_mode", SetFSMState)
        state_machine_switch_node(new_state)

    def handle_ToF(self, data):
        """Store the fetched ToF measurement. For out of range measurements, the provided sensor value is >8.0 and the actual value is set to 0.0 instead"""
        detected_distance = 0.0 if data.range > 1.0 else data.range
        self.tof = detected_distance

    def handle_FSM(self, data):
        """Store the fetched FSM state"""
        self.state = data.state
        if self.state == 'WAITING':
            if len(node.route) > 0:
                old_state = self.route.pop(0)
                self.loginfo(f"Popped {old_state} from the route, it's now {self.route}")
            else:
                rospy.loginfo("Route is empty, no command to pop")

        

    def handle_pose(self, data):
        """Store the fetched odometry pose"""
        self.pose['x'] = data.x
        self.pose['y'] = data.y
        self.pose['theta'] = data.theta


if __name__ == '__main__':
    # create the node
    node = HTTPDuckieServer(node_name='http_duckie_server')  # run node
    node.start_server()

    # keep spinning
    rospy.spin()
