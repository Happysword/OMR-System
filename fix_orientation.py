import numpy as np
import cv2
import math


# Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
def fix_orientation(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.bool:
        img = img.astype(np.uint8) * 255

    cropped_borders = __crop_borders(img)
    (height, width) = cropped_borders.shape
    binary_image = __binarize_with_canny(cropped_borders)

    angle = __get_rotation_angle(binary_image)

    binary_rotated = __rotate_image(binary_image, angle)
    transformation_matrix = get_perspective_transformation_matrix(binary_rotated)
    binary_perspective = cv2.warpPerspective(binary_rotated, transformation_matrix,
                                             (width, height), borderMode=cv2.BORDER_REPLICATE)

    img_rotated = __rotate_image(cropped_borders, angle)
    img_perspective = cv2.warpPerspective(img_rotated, transformation_matrix,
                                          (width, height), borderMode=cv2.BORDER_REPLICATE)

    x1, y1, x2, y2 = __get_cropping_rectangle(binary_perspective)
    return img_perspective[y1:y2, x1:x2]


def __rotate_image(img: np.ndarray, angle_in_degrees) -> np.ndarray:
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_in_degrees, scale=1)
    return cv2.warpAffine(img, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    # return cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=0)


def __crop_borders(img: np.ndarray, percentage: float = 0.01) -> np.ndarray:
    (h, w) = img.shape[:2]
    delta_h = int(percentage * h)
    delta_w = int(percentage * w)
    return img[delta_h:h - delta_h, delta_w:w - delta_w]


def __get_line_length(line):
    x1, y1, x2, y2 = np.array(line.copy()).flatten()
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Get Line's slope angle in degrees
def __get_line_angle(line):
    x1, y1, x2, y2 = np.array(line.copy()).flatten()
    x = x2 - x1
    y = y2 - y1
    return math.degrees(math.atan2(y, x))


def get_bounding_lines(hull_points) -> np.ndarray:
    np.append(hull_points, [hull_points[0]], axis=0)

    lines = [[hull_points[0], hull_points[1]]]
    old_angle = __get_line_angle(lines[0])

    for i in range(2, len(hull_points)):
        line = [hull_points[i - 1], hull_points[i]]
        new_angle = __get_line_angle(line)
        if abs(old_angle - new_angle) <= 15:
            lines[-1][1] = hull_points[i]
        else:
            lines.append(line)
        old_angle = __get_line_angle(lines[-1])

    lines.sort(key=__get_line_length, reverse=True)
    return __sort_boundary_lines(lines[0:4])


# Sort boundary lines to be : left, top, right, bottom
# Left have min sum of x , right have max sum of x
# Top have min sum of y , Bottom have max sum of y
def __sort_boundary_lines(boundary_lines: list):
    boundary_lines = np.array(boundary_lines)
    sum: np.ndarray = boundary_lines.sum(axis=1)
    sum = sum.reshape((4, 2))

    min_x_index = np.argmin(sum, axis=0)[0]
    min_y_index = np.argmin(sum, axis=0)[1]
    max_x_index = np.argmax(sum, axis=0)[0]
    max_y_index = np.argmax(sum, axis=0)[1]

    rect = np.zeros_like(boundary_lines)
    rect[0] = boundary_lines[min_x_index]
    rect[1] = boundary_lines[min_y_index]
    rect[2] = boundary_lines[max_x_index]
    rect[3] = boundary_lines[max_y_index]

    return rect


# Line is defined by s (starting point) , e (ending point)
def __get_intersection(line1, line2):
    s1, e1 = np.array(line1, dtype=float).reshape((2, 2))
    s2, e2 = np.array(line2, dtype=float).reshape((2, 2))

    x = s2 - s1
    direction1 = e1 - s1
    direction2 = e2 - s2

    cross_product = direction1[0] * direction2[1] - direction1[1] * direction2[0]

    # if lines are parallel there is no intersection
    if abs(cross_product) < 0.0000001:
        raise ArithmeticError

    t1 = (x[0] * direction2[1] - x[1] * direction2[0]) / cross_product
    return s1 + direction1 * t1


def get_perspective_transformation_matrix(img: np.ndarray) -> np.ndarray:
    all_points = cv2.findNonZero(img)
    hull_points: np.ndarray = cv2.convexHull(all_points)

    bounding_lines = get_bounding_lines(hull_points)

    intersection_points = np.zeros((4, 2), dtype=np.float32)
    for i, line in enumerate(bounding_lines):
        next_line = bounding_lines[(i + 1) % len(bounding_lines)]
        intersection = __get_intersection(line, next_line)
        intersection_points[i] = intersection

    x, y, w, h = cv2.boundingRect(hull_points)

    rectangle_points = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    transformation_matrix = cv2.getPerspectiveTransform(intersection_points, rectangle_points)
    return transformation_matrix


def __get_cropping_rectangle(binary_image: np.ndarray):
    x_fix = binary_image.shape[1] // 25
    y_fix = binary_image.shape[0] // 25
    all_points = cv2.findNonZero(binary_image)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x - x_fix), (y - y_fix), (x + x_fix + w), (y + y_fix + h)


def __get_hough_lines(binary_image: np.ndarray):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = max(binary_image.shape[0],
                    binary_image.shape[1]) // 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 2 * max(binary_image.shape[0], binary_image.shape[1]) // 3
    max_line_gap = min_line_length / 10  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines


def __get_rotation_angle(binarized_image: np.ndarray):
    lines_endpoints = __get_hough_lines(binarized_image)

    data_type = [('length', float), ('angle', float)]
    lines_properties = np.zeros(len(lines_endpoints), dtype=data_type)

    for i, line in enumerate(lines_endpoints):
        lines_properties[i]['length'] = __get_line_length(line)
        lines_properties[i]['angle'] = __get_line_angle(line)

    lines_properties[::-1].sort(order='length')  # sort in descending order according to length field
    return np.average(lines_properties[0:5]['angle'])


def __binarize_with_canny(img: np.ndarray):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 3)

    low_threshold = 70
    high_threshold = 150
    edges = cv2.Canny(img, low_threshold, high_threshold)

    return cv2.dilate(edges, (9, 9))
