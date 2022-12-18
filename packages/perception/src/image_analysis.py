import cv2
import numpy as np

import hough_line_transformation
import line_processing as lp
import preprocessing


def detect_duckie_bot_board(image, color_dict=None):
    if color_dict is None:
        color_dict = preprocessing.extract_colors(image)

    """This method tries to detect the duckiebot board in the image."""
    white_image_parts = color_dict[preprocessing.Hues.WHITE]
    white_image_parts = cv2.morphologyEx(white_image_parts, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(white_image_parts)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (192, 222, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return len(keypoints) > 15, im_with_keypoints


def detect_redline(image, color_dict=None, max_angle=np.pi / 3, red_top_crop=0.75, length_threshold=300, debug=False):
    """This method tries to create a horizontal red line directly in front of the duckiebot."""
    if color_dict is None:
        color_dict = preprocessing.extract_colors(image)

    red_average_angle, lines, angles = lp.line_angles_from_image(color_dict, preprocessing.Hues.RED, lines_used=1000, top_crop=red_top_crop)

    if lines is None or angles is None:
        return False, None, 0

    # Take the absolute value to make sure slightly different vertical lines do not create a horizontal mean.
    # This is caused by -90 on one side of a vertical line and 90 on the other.
    angles = np.where(angles > np.pi / 2, angles - np.pi, angles)
    angles = np.abs(angles)

    if debug:
        hough_line_transformation.write_euclid_hough_lines(image, lines, (0, 0, 255))

    lengths = np.linalg.norm(lines[:, 0:2] - lines[:, 2:4], axis=1)
    length = np.sum(lengths)

    if length <= length_threshold:
        return False, None, length

    potential_angles = angles[angles < max_angle]
    potential_lengths = lengths[angles < max_angle]

    sorted_angle_indexes = np.argsort(potential_angles)
    sorted_angles = potential_angles[sorted_angle_indexes]
    sorted_lengths = potential_lengths[sorted_angle_indexes]

    cutoff_index = np.array([np.sum(np.square(sorted_angles - angle)) for angle in sorted_angles]).argmin()

    len_one = np.sum(sorted_lengths[:cutoff_index])
    len_two = np.sum(sorted_lengths[cutoff_index:])

    if len_one > len_two:
        average_angle = np.mean(sorted_angles[:cutoff_index])
    else:
        average_angle = np.mean(sorted_angles[cutoff_index:])

    return True, average_angle, length


def detect_lane(img, color_dict=None, look_ahead=0.75):
    h, w = img.shape[:2]

    if color_dict is None:
        color_dict = preprocessing.extract_colors(img)

    white_angle, w_lines, _ = lp.line_angles_from_image(color_dict, preprocessing.Hues.WHITE, 10, top_crop=look_ahead)
    yellow_angle, y_lines, _ = lp.line_angles_from_image(color_dict, preprocessing.Hues.YELLOW, 10, top_crop=look_ahead)

    if y_lines is None or len(y_lines) == 0:
        y_lines = np.array([[0, h - 1, 0, h - 11], ])
        yellow_angle = np.pi / 2

    if w_lines is None or len(w_lines) == 0:
        w_lines = np.array([[w - 1, h - 1, w - 1, h - 11], ])
        white_angle = np.pi / 2

    mid_point = lp.get_midpoint_from_lines(y_lines, w_lines, debug_img=img)
    return mid_point, (white_angle, yellow_angle), img


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    while (cv2.waitKey(10) & 0xFF) != ord('q'):
        ret, frame = cam.read()
        if ret:
            color_dict = preprocessing.extract_colors(frame, top_crop=0.0)
            _, _, debug = detect_lane(frame, color_dict=color_dict)
            bot, debug = detect_duckie_bot_board(debug, color_dict=color_dict)
            print(f"Redline: {detect_redline(frame, color_dict=color_dict)}, Duckiebot: {bot}")
            cv2.imshow("Debug", debug)
    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
