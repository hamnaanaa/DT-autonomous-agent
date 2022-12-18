import cv2
import numpy as np


def hough_lines_euclid(edge_image, thresh=30):
    """This method will use a canny edge detector followed by a hough transformation to extract all lines from an
    image.
    Returns the lines in form of a numpy array."""

    edge_image = cv2.Canny(edge_image, 100, 200)
    lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi / 180, threshold=thresh, minLineLength=5, maxLineGap=5)

    # Automatically flatten the lines.
    return None if lines is None else np.array(lines)[:, 0]


def write_euclid_hough_lines(image, lines, color=(0, 0, 255)):
    if lines is None:
        return

    for line in lines:
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color, 2)

    return image
