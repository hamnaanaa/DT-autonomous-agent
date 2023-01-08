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

# import CameraConfig
# import image_analysis
# import preprocessing

# import torch
# import torchvision
# from torchvision import transforms
# from segmentation_model import DTSegmentationNetwork

import io
from PIL import Image

# import rospkg


class LanePerceptionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LanePerceptionNode, self).__init__(node_name=node_name,
                                                 node_type=NodeType.BEHAVIOR)
        self.veh_name = rospy.get_namespace().strip("/")

        # self.bridge = CvBridge()

        # self.camera_config = CameraConfig.CameraConfig.from_file(f"/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml")

        # params
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/gamma", 0.9)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/look_ahead", 0.75)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_angle", np.pi / 3)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_top_crop", 0.75)
        rospy.set_param(f"/{self.veh_name}/lane_perception_node/red_line_length_threshold", 500)

        # sub, pub
        self.mode_sub = rospy.Subscriber(f"/{self.veh_name}/state_machine_node/mode", FSMState, self.cb_on_mode_change)
        # self.im_sub = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_img, queue_size=1,
        #                                buff_size=1000000)
        # self.lane_det_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/lane_det", LaneDetection, queue_size=1)
        self.line_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/lines/compressed", CompressedImage, queue_size=1)
        # self.red_line_pub = rospy.Publisher(f"/{self.veh_name}/lane_perception_node/red_lines/compressed", CompressedImage, queue_size=1)

        change_mode_srv = f"/{self.veh_name}/state_machine_node/change_mode"
        self.mode_srv = rospy.ServiceProxy(change_mode_srv, SetFSMState)

        # self.last_red = rospy.get_time()
        # self.red_active = False

        # self.rect_mat = None

        self.image = None
        self.loginfo("Initialized")
        
        # self.loginfo(f"Use pytorch of version: {torch.__version__}")
        # self.loginfo(f"Use torchvision of version: {torchvision.__version__}")
        
        # hparams = {
        #     # --- Model ---
        #     # | Model hyperparameters
        #     'num_classes': 2,
        #     # | Optimization hyperparameters
        #     "learning_rate": 0.0625,
        #     "weight_decay": 0.000000625,
        #     "lr_decay": 0.25,
            
        #     # --- Dataloader (Hardware-specific) ---
        #     "batch_size": 16,
        #     "num_workers": 4,
        # }
        # self.loginfo("[HAM] Initialize model...")
        # self.model = DTSegmentationNetwork(hparams)
        # self.loginfo("[HAM] Initialized empty model")
        # rospack = rospkg.RosPack()
        # path_to_model = rospack.get_path('perception') + "/src/model_v7_0_086_state_dict.pt"
        # self.loginfo(f"[HAM] Load model from file: {path_to_model}...")
        # self.model.load_state_dict(torch.load(path_to_model))
        # self.loginfo("[HAM] Loaded model with state dict!")
        
        # self.avg_process_time = 0
        

    def cb_on_mode_change(self, msg):
        if msg.state == "LANE_FOLLOWING":
            self.red_active = True


    def _pred_to_img(self, label_img, color_map):
        # Convert the label image to a color image
        label_img = label_img.cpu().numpy()
        color_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
        for label, color in color_map.items():
            color_img[label_img == label] = color
        
        # Convert the color image to a PIL image
        color_img = Image.fromarray(color_img)
        return color_img
        


    def cb_img(self, message):
        # if self.camera_config is None or self.camera_header is None:
        #    return
        if not hasattr(self, "model"):
            self.loginfo("[HAM] Model is not initialized yet!")
            return
        
        # self.image = Image.open(io.BytesIO(message.data)).convert("RGB")
        
        # send the image 
        
        # start_time = rospy.Time.now()
        # # Convert the image to a PIL image and then back to bytes to test the conversion
        # image = Image.open(io.BytesIO(message.data)).convert("RGB")
        # time_to_pil = rospy.Time.now()
        # # self.loginfo(f"[HAM] Converting image to PIL took {(rospy.Time.now() - delta_time).to_sec()} seconds")

        # image = transforms.ToTensor()(image)
        # time_to_tensor = rospy.Time.now()
        # # self.loginfo(f"[HAM] Tensor image size: {image.size()}")
        
        # image = torch.argmax(self.model(image.unsqueeze(0)), dim=1)[0]
        # time_to_pred = rospy.Time.now()
        # image = self._pred_to_img(image, {0: [0, 0, 0], 1: [0, 255, 0]})
        # time_to_compressed_img = rospy.Time.now()
        # # self.loginfo(f"[HAM] Prediction size: {image.shape}. Create a CompressedImage from it. Here are the first 10 values: {image[:10]}")
        
        # # Convert the result to a CompressedImage to publish on ROS topic
        # msg = CompressedImage()
        # msg.header.stamp = rospy.Time.now()
        # msg.format = "jpeg"
        # buf = io.BytesIO()
        # image.save(buf, format="JPEG")
        # msg.data = buf.getvalue()
        # time_to_save = rospy.Time.now()
        
        # process_time = (rospy.Time.now() - start_time).to_sec()
        # self.avg_process_time = process_time if self.avg_process_time == 0 else (self.avg_process_time * 0.75 + process_time * 0.25)
        # self.loginfo(f"""
        # [HAM] Created a message to publish. Here are the times summary:
        # - Received image at: {start_time.to_sec()}
        # - Process time (curr/avg): {process_time}/{self.avg_process_time}
        # - Time to PIL: {(time_to_pil - start_time).to_sec()}
        # - Time to Tensor: {(time_to_tensor - time_to_pil).to_sec()}
        # - Time to Prediction: {(time_to_pred - time_to_tensor).to_sec()}
        # - Time to CompressedImage: {(time_to_compressed_img - time_to_pred).to_sec()}
        # - Time to Save: {(time_to_save - time_to_compressed_img).to_sec()}
        # """)
        # self.line_pub.publish(msg)


if __name__ == '__main__':
    # Initialize the node
    node = LanePerceptionNode(node_name='lane_perception_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
