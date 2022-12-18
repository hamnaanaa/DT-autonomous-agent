from enum import Enum

import cv2
import numpy as np


# HSV ranges
# First pair is Hue, second saturation, third value.
class Hues(Enum):
    RED = 0
    YELLOW = 1
    WHITE = 2


def gamma_correction(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def crop_camera_image(image, top_crop=0.75, bottom_crop=0.04, left_crop=0.0, right_crop=0.0):
    """This method crops the image to only include the road that interests us."""
    cropped = image.copy()
    height, width = cropped.shape[:2]
    cropped[: int(height * top_crop), :] = 0
    cropped[height - 1 - int(height * bottom_crop):, :] = 0
    cropped[:, : int(width * left_crop)] = 0
    cropped[:, width - 1 - int(width * right_crop):] = 0

    target_height = int(0.66 * height)
    target_width = int(0.66 * width)
    left_triangle = np.array([(0, 0), (0, target_height), (target_width, 0)])
    right_triangle = np.array([(width - 1, 0), (width - 1, target_height), (width - target_width, 0)])

    cv2.drawContours(cropped, [left_triangle], 0, (0, 0, 0), -1)
    cv2.drawContours(cropped, [right_triangle], 0, (0, 0, 0), -1)

    return cropped


def extract_colors_new(image):
    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_REFLECT_101)

    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_channel, red_green, yellow_blue = cv2.split(img_lab)

    _, color_luminosity = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY)
    _, color_low_luminosity = cv2.threshold(l_channel, 60, 255, cv2.THRESH_BINARY)

    # Extract yellow:
    _, yellow_mask = cv2.threshold(yellow_blue, 130, 255, cv2.THRESH_BINARY)
    yellow_mask = cv2.bitwise_and(yellow_mask, color_luminosity)

    _, inv_red_mask = cv2.threshold(red_green, 130, 255, cv2.THRESH_BINARY_INV)
    yellow_mask = cv2.bitwise_and(yellow_mask, inv_red_mask)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    yellow_dilated = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)

    # Extract white:
    _, white_mask = cv2.threshold(l_channel, 145, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_or(extract_hsv_color(img_hsv, 0, 359, 0, 25, 50, 100), white_mask)
    white_mask = cv2.bitwise_and(cv2.bitwise_not(yellow_dilated), white_mask)

    # Extract red:
    _, red_mask = cv2.threshold(red_green, 140, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(red_mask, color_luminosity)
    red_mask = cv2.bitwise_or(red_mask, cv2.threshold(red_green, 150, 255, cv2.THRESH_BINARY)[1])
    red_mask = cv2.bitwise_and(red_mask, color_low_luminosity)
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(yellow_dilated))
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(cv2.threshold(yellow_blue, 120, 255, cv2.THRESH_BINARY_INV)[1]))
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(white_mask))

    return yellow_mask, white_mask, red_mask


def extract_colors(image, top_crop=0.0):
    """
    This method uses the hsv format to extract a certain, continuous color interval from an image.
    The input ranges are [0, 0, 0] to [359, 100, 100]

    Hues is a list of pairs (min_hue, max_hue)
    """
    image = crop_camera_image(image, top_crop=top_crop)
    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_REFLECT_101)

    colors = {Hues.RED: [], Hues.YELLOW: [], Hues.WHITE: []}

    colors[Hues.YELLOW], colors[Hues.WHITE], colors[Hues.RED] = extract_colors_new(image)

    for hue, mask in colors.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        colors[hue] = mask

    return colors


def extract_hsv_color(image, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    # HSV is normally in the range [359, 100, 100], but CV2 uses [179, 255, 255]
    lower_mask = np.array([min_hue // 2, round(min_saturation * 2.55), round(min_value * 2.55)])
    upper_mask = np.array([max_hue // 2, round(max_saturation * 2.55), round(max_value * 2.55)])
    return cv2.inRange(image, lower_mask, upper_mask)
