###############################################################################
# Duckietown - Project intnav ETH
# Author: Simon Schaefer
# Load and make availabe camera configuration data from configuration file
# or camera configuration message string.
# K - intrinsics.
# H - homography.
# R - rectification matrix.
# P - projjection matrix.
###############################################################################
import os.path

import cv2
import numpy as np
import yaml
from numpy import asarray, reshape


class CameraConfig:

    def __init__(self, K, D, H, dim, P, R):
        # Initialize camera properties.
        self.K = reshape(asarray(K), (3,3))
        self.D = asarray(D)
        self.width = dim[0]
        self.height = dim[1]
        self.P = reshape(asarray(P), (3,4))
        self.R = reshape(asarray(R), (3,3))
        self.H = None
        if not H is None:
            self.H = reshape(asarray(H), (3,3))
        # Initialize camera model and undistortion model.
        mapx = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        mapy = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (self.width, self.height),
            cv2.CV_32FC1, mapx, mapy)

    @classmethod
    def from_file(cls, calibration_file_path="./camera.yaml"):
        ''' Load camera intrinsics and distortion parameter from yaml file,
        which is stored in data directory by default. '''
        params = {}
        try:
            config_file = os.path.dirname(os.path.realpath(__file__))
            config_file = os.path.join(config_file, calibration_file_path)
            with open(config_file, 'r') as stream:
                params = yaml.safe_load(stream)
        except (IOError, yaml.YAMLError):
            raise IOError("Unknown or invalid parameters file !")

        # Assign parameters.
        K = params['camera_matrix']['data']
        dist_params = params['distortion_coefficients']['data']
        img_width = params['image_width']
        img_height = params['image_height']
        projection_matrix = params['projection_matrix']['data']
        rect_matrix = params['rectification_matrix']['data']
        homography = None
        return CameraConfig(K, dist_params, homography, (img_width, img_height),
                            projection_matrix, rect_matrix)

    def convert_pixel_to_world(self, pixel_coords):
        ''' Convert pixel to world coordinates using homography.
        @param[in]  pixel_coords        (u,v) pixel coordinates. '''
        pixel = np.array([pixel_coords[1], pixel_coords[0], 1])
        return np.matmul(self.H, pixel)

    def rectify_image(self, image):
        image_rectified = np.zeros(np.shape(image))
        return cv2.remap(image, self._mapx, self._mapy, cv2.INTER_CUBIC, image_rectified)
