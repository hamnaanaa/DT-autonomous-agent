import math

import cv2
import numpy as np

import hough_line_transformation as hough
import preprocessing


def sort_lines_lowest_position(lines):
    """Sorts the lines so that the point with the lowest y position on the picture is first in the sorted list."""
    return np.array(sorted(lines, key=lambda line: max(line[1], line[3]), reverse=True)) if lines is not None else None


def calculate_angle(line):
    """Expects a line in the format [x1, y1, x2, y2] and calculates the angle of the line."""
    return math.atan2((line[3] - line[1]), (line[2] - line[0]))


def calculate_angles_from_lines(lines):
    """Expects lines as returned by hough_lines_euclid."""
    if lines is None:
        return None

    return np.arctan2(lines[:, 3] - lines[:, 1],
                      lines[:, 2] - lines[:, 0])


def human_readable_angle(angle):
    return math.degrees(angle)


def human_readable_angles(angles):
    """Prints the angles of hough transform in human-readable form."""
    return np.degrees(angles)


def line_angles_from_image(color_dict, hue, lines_used=3, top_crop=0.75, bottom_crop=0.0, left_crop=0.0,
                           right_crop=0.0,
                           debug=False, debug_title="0_DEBUG"):
    """This calculates all the lines found in the image after cropping, sorted from bottommost start to upmost start in
    the picture, with their respective angles and the mean angle of all lines.

     The method returns a triple (mean: number, lines: sorted list, angles: np array)
     """
    color_image = preprocessing.crop_camera_image(color_dict[hue], top_crop, bottom_crop, left_crop, right_crop)

    if debug:
        cv2.imwrite(f"./test_assets/{debug_title}_preprocessed.png", color_image)

    lines = hough.hough_lines_euclid(color_image)

    if lines is None:
        return None, None, None

    lines = sort_lines_lowest_position(lines)[:lines_used]
    angles = calculate_angles_from_lines(lines)

    # Take the absolute value to make sure slightly different vertical lines do not create a horizontal mean.
    # This is caused by -90 on one side of a vertical line and 90 on the other.
    angles = np.where(angles < 0, angles + np.pi, angles)
    angles = np.pi - angles
    mean = np.mean(angles) if len(angles) > 0 else None

    return mean, lines, angles


def get_midpoint_from_lines(yellow_lines, white_lines, debug_img=None):
    yellow_pts = np.reshape(yellow_lines, (-1, 2)).astype(int)
    white_pts = np.reshape(white_lines, (-1, 2)).astype(int)

    mid_points = []

    for start_pt in yellow_pts:
        # Ignore all points left of the yellow line.
        possible_end_pts = white_pts[start_pt[0] < white_pts[:, 0]]

        if len(possible_end_pts) == 0:
            continue

        # Calculate the y-distance to the yellow line point for every white point.
        y_dists = np.abs(possible_end_pts[:, 1] - start_pt[1])

        # Use the point with the lowest y-distance to calculate the middle.
        min_dist_point = possible_end_pts[np.argmin(y_dists)]

        # Take the mean of the start and end point on the yellow and white lines to get the midpoint.
        mid_point = ((start_pt + min_dist_point) / 2.0).astype(int)

        if not np.isnan(mid_point).any():
            mid_points.append(mid_point)

            if debug_img is not None:
                cv2.circle(debug_img, tuple(min_dist_point), 3, (0, 255, 0), 1)
                cv2.circle(debug_img, tuple(start_pt), 3, (255, 0, 0), 1)
                cv2.circle(debug_img, tuple(mid_point), 3, (0, 0, 255), 1)

    if len(mid_points) == 0:
        return None

    mid_points = np.array(mid_points)

    if len(mid_points) == 0:
        return None

    std_mid_points = mid_points
    mean = np.mean(std_mid_points, 0)
    dists = np.linalg.norm(std_mid_points - mean, axis=1)
    std_mid_points = std_mid_points[dists < np.std(dists)]

    # If we have no better data, use the outliers.
    if len(std_mid_points) != 0 and not (np.isnan(std_mid_points)).any():
        mid_points = std_mid_points

    mid_point = np.mean(mid_points, axis=0)

    if mid_point is not None:
        cv2.circle(debug_img, tuple(mid_point.astype(int)), 5, (255, 255, 255), 2)

    if np.isnan(mid_point).any():
        return None

    return mid_point
