# import the necessary packages
import cv2, os
import numpy as np


def find_sorted_rectangles(contours):
    """ Finds the biggest rectangle in a list of rectangles.

    Args:
        contours (list): A list of rectangles.

    Returns:
        list: A list of rectangles sorted by area.
    """
    rectangles = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # print(len(approx)) # we actually care if we have 4 edges here ( it's a rectangle !! )
        if len(approx) == 4 and area > 1000:
            rectangles.append(c)

    # print(len(rectangles))
    rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)
    # print(rectangles)
    return rectangles


def detect_corners(rectangle):
    peri = cv2.arcLength(rectangle, True)
    approx = cv2.approxPolyDP(rectangle, 0.04 * peri, True)

    return approx


def rearrange(points):
    # the shape of the question_box rectangle now is (4, 1, 2), so we
    # need to remove the 1 dimension as we don't need it.
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    addition = points.sum(1)
    subtract = np.diff(points, axis=1)
    new_points[0] = points[np.argmin(addition)]
    new_points[3] = points[np.argmax(addition)]
    new_points[1] = points[np.argmin(subtract)]
    new_points[2] = points[np.argmax(subtract)]

    return new_points


def get_contour_precedence(contour, cols):
    """ Get the contour's area and then divide by the number of columns in the image.

    Args:
        contour (numpy.ndarray): Contour.
        cols (int): Number of columns in the image.

    Returns:
        float: Contour's area divided by the number of columns in the image.
    """
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def sort_contours(contours, method="left-to-right"):
    """ Sort the contours from left-to-right or top-to-bottom.

    Args:
        contours (list): A list of rectangles.
        method (str, optional): Contour's order. Defaults to "left-to-right".

    Returns:
        list: A list of rectangles sorted from left-to-right or top-to-bottom.
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(
        *sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse)
    )
    # return the list of sorted contours and bounding boxes
    return contours, bounding_boxes


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """ Resize an image.

    Args:
        image (numpy.ndarray): Image.
        width (int, optional): Width. Defaults to None.
        height (int, optional): Height. Defaults to None.
        inter (int, optional): Interpolation. Defaults to cv2.INTER_AREA.

    Returns:
        numpy.ndarray: Resized image.
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def rename_image_with_grades(q, grade):
    """ Rename the image with the grade.

    Args:
        q (str): Question.
        grade (str): Grade.
    """
    open_brackets = "("
    close_brackets = ")"
    str_grade = str(grade)
    rename = str(open_brackets + str_grade + close_brackets)
    prev_name = str(q.path)
    image_name = q.name.split(".")
    first_part = str(image_name[0] + rename)
    print(first_part)
    ext = ".jpg"
    str_ext = str(ext)
    new_name1 = str(first_part + str_ext)
    if rename not in q.name:
        os.rename(prev_name, new_name1)
