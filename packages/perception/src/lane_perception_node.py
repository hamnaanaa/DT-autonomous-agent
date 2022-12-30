#!/usr/bin/env python3

# ALWAYS import cv2 before cvbridge (even when not otherwise needed):
# https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
# noinspection PyUnresolvedReferences
import cv2
import numpy as np
import rospy
from custom_msgs.msg import LaneDetection
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.srv import SetValue, SetFSMState
from duckietown_msgs.msg import FSMState

import CameraConfig
import image_analysis
import preprocessing

import torch
import torchvision
from torchvision import transforms
from segmentation_model import DTSegmentationNetwork

import io
from PIL import Image

import rospkg


class LanePerceptionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LanePerceptionNode, self).__init__(node_name=node_name,
                                                 node_type=NodeType.BEHAVIOR)
        self.veh_name = rospy.get_namespace().strip("/")

        self.bridge = CvBridge()

        self.camera_config = CameraConfig.CameraConfig.from_file(f"/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml")

        # params
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/gamma", 0.9)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/look_ahead", 0.75)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_angle", np.pi / 3)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_top_crop", 0.75)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_line_length_threshold", 500)

        # sub, pub
        self.mode_sub = rospy.Subscriber(f"/{self.veh_name}/state_machine_node/mode", FSMState, self.cb_on_mode_change)
        self.im_sub = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_img, queue_size=1,
                                       buff_size=1000000)
        self.lane_det_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/lane_det", LaneDetection, queue_size=1)
        self.line_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/lines/compressed", CompressedImage, queue_size=1)
        # self.red_line_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/red_lines/compressed", CompressedImage, queue_size=1)

        change_mode_srv = f"/{self.veh_name}/state_machine_node/change_mode"
        self.mode_srv = rospy.ServiceProxy(change_mode_srv, SetFSMState)

        self.last_red = rospy.get_time()
        self.red_active = False

        self.rect_mat = None

        self.loginfo("Initialized")
        
        self.loginfo(f"Use pytorch of version: {torch.__version__}")
        self.loginfo(f"Use torchvision of version: {torchvision.__version__}")
        
        hparams = {
            # --- Model ---
            # | Model hyperparameters
            'num_classes': 2,
            # | Optimization hyperparameters
            "learning_rate": 0.0625,
            "weight_decay": 0.000000625,
            "lr_decay": 0.25,
            
            # --- Dataloader (Hardware-specific) ---
            "batch_size": 16,
            "num_workers": 4,
        }
        self.loginfo("[HAM] Initialize model...")
        self.model = DTSegmentationNetwork(hparams)
        self.loginfo("[HAM] Initialized empty model")
        rospack = rospkg.RosPack()
        path_to_model = rospack.get_path('perception') + "/src/model_v7_0_086_state_dict.pt"
        self.loginfo(f"[HAM] Load model from file: {path_to_model}...")
        self.model.load_state_dict(torch.load(path_to_model))
        self.loginfo("[HAM] Loaded model with state dict!")
        

    def cb_on_mode_change(self, msg):
        if msg.state == "LANE_FOLLOWING":
            self.red_active = True


    def cb_img(self, message):
        # if self.camera_config is None or self.camera_header is None:
        #    return
        if not self.model:
            self.loginfo("[HAM] Model is not initialized yet!")
            return
        
        # Convert the image to a PIL image and then back to bytes to test the conversion
        image = Image.open(io.BytesIO(message.data)).convert("RGB")
        self.loginfo(f"[HAM] Image received and converted to PIL Image with size: {image.size}")
        
        # RAW back -> And then back to bytes
        # buf = io.BytesIO()
        # image.save(buf, format="JPEG")

        
        
        image = transforms.ToTensor()(image)
        # self.loginfo(f"[HAM] Tensor image size: {image.size()}")
        
        # result = torch.argmax(self.model(image.unsqueeze(0)), dim=1)[0]
        # self.loginfo(f"[HAM] Result size: {result.size()}. Create a CompressedImage...")
        
        # Convert the result to a CompressedImage to publish on ROS topic
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        buf = io.BytesIO()
        transforms.ToPILImage()(image).save(buf, format="JPEG")
        msg.data = buf.getvalue()
        
        
        self.loginfo(f"[HAM] Created a message to publish.")
        self.line_pub.publish(msg)
        
        
        
        
        
        # image = self.bridge.compressed_imgmsg_to_cv2(message, desired_encoding="passthrough")

        # gamma = rospy.get_param(f"/{self.veh_name}/lane_perception_node/gamma")
        # image = preprocessing.gamma_correction(image, gamma)
        # rect_image = self.camera_config.rectify_image(image)
        # rect_color_dict = preprocessing.extract_colors(rect_image)

        # h, w = image.shape[:2]

        # look_ahead = rospy.get_param(f"/{self.veh_name}/lane_perception_node/look_ahead")
        # color_dict = preprocessing.extract_colors(image)

        # mid_point, _, line_img = image_analysis.detect_lane(image, color_dict=color_dict, look_ahead=look_ahead)

        # red_angle = rospy.get_param(f"/{self.veh_name}/lane_perception_node/red_angle")
        # red_top_crop = rospy.get_param(f"/{self.veh_name}/lane_perception_node/red_top_crop")
        # red_line_length_threshold = rospy.get_param(f"/{self.veh_name}/lane_perception_node/red_line_length_threshold")

        # redline_detected, redline_angle, length = image_analysis.detect_redline(rect_image, color_dict=rect_color_dict, max_angle=red_angle,
        #                                                                         red_top_crop=red_top_crop,
        #                                                                         length_threshold=red_line_length_threshold, debug=True)

        # #self.loginfo(f"Redline: {redline_detected} with angle {redline_angle} and length {length}")
        # if redline_angle is not None and self.red_active:
        #     #self.red_line_srv(redline_angle) # TODO angle is not signed!
        #     self.mode_srv("WAITING")
        #     self.red_active = False

        # line_img_msg = self.bridge.cv2_to_compressed_imgmsg(line_img)
        # red_line_img_msg = self.bridge.cv2_to_compressed_imgmsg(rect_image)

        # det = LaneDetection()
        # if mid_point is not None:
        #     det.mid_point_h = mid_point[0] / w
        #     det.mid_point_v = mid_point[1] / h

        # self.lane_det_pub.publish(det)
        # self.line_pub.publish(pred_image)
        # self.red_line_pub.publish(red_line_img_msg)


if __name__ == '__main__':
    # Initialize the node
    node = LanePerceptionNode(node_name='lane_perception_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
