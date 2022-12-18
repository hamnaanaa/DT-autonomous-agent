import cv2
import numpy as np

import CameraConfig
import hough_line_transformation
import image_analysis
import line_processing
import preprocessing

if __name__ == "__main__":
    paths = [f"./test_assets/track_test_{i}.png" for i in range(0, 20)]

    test_images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
    camera_config = CameraConfig.CameraConfig.from_file()

    for index, test_image in enumerate(test_images):
        test_image = cv2.resize(test_image, (640, 480))
        test_image = camera_config.rectify_image(test_image)

        gamma = 0.9
        test_image = preprocessing.gamma_correction(test_image, gamma)

        h, w = test_image.shape[:2]

        look_ahead = 0.75
        angle_part = 0.25

        color_dict = preprocessing.extract_colors(test_image, look_ahead)

        _, red, _ = line_processing.line_angles_from_image(color_dict, preprocessing.Hues.RED, 100, top_crop=look_ahead,
                                                           debug=True, debug_title="{}_RED".format(index))

        hough_line_transformation.write_euclid_hough_lines(test_image, red, (0, 0, 255))

        _, white, _ = line_processing.line_angles_from_image(color_dict, preprocessing.Hues.WHITE, 10, debug=True, top_crop=look_ahead,
                                                             debug_title="{}_WHITE".format(index))

        hough_line_transformation.write_euclid_hough_lines(test_image, white, (255, 255, 255))

        _, yellow, _ = line_processing.line_angles_from_image(color_dict, preprocessing.Hues.YELLOW, 10, debug=True, top_crop=look_ahead,
                                                              debug_title="{}_YELLOW".format(index))

        hough_line_transformation.write_euclid_hough_lines(test_image, yellow, (0, 255, 255))

        _, _, img = image_analysis.detect_lane(test_image, color_dict=color_dict, look_ahead=look_ahead)

        red_line_found, red_line_angle, length = image_analysis.detect_redline(test_image, color_dict=color_dict)
        if red_line_angle is not None:
            red_line_angle = np.degrees(red_line_angle)

        print(f"Image: {index} | Horizontal red line found: {red_line_found}, with angle {red_line_angle} degrees and length {length}.")
        cv2.imwrite(f"./test_assets/{index}_lanes.png", test_image)
