import numpy as np
import cv2
import math

__DEBUG__ = False


# Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
def fix_orientation(img: np.ndarray, debug=False) -> np.ndarray:
    global __DEBUG__
    __DEBUG__ = debug
    if img.dtype == np.bool:
        img = img.astype(np.uint8) * 255

    __debug_show_image(img)

    img = cv2.bitwise_not(img)
    img = remove_noise(img)
    img = __crop_image(img)

    angle_hough = __get_rotation_angle_hough(img)
    angle = (abs(__get_rotation_angle(img)) + abs(angle_hough)) / 2
    angle = -angle if (angle_hough < 0) else angle

    if __DEBUG__:
        print(angle)

    img_rotated = __rotate_image(img, angle)
    __debug_show_image(img_rotated)

    img_rotated = __crop_image(img_rotated)
    (height, width) = img_rotated.shape
    __debug_show_image(img_rotated)

    transformation_matrix = get_perspective_transformation_matrix(img_rotated)
    img_perspective = cv2.warpPerspective(img_rotated, transformation_matrix, (width, height),
                                          flags=cv2.WARP_FILL_OUTLIERS)

    __debug_show_image(img_perspective)

    return cv2.bitwise_not(img_perspective)


def remove_noise(binary_image: np.ndarray) -> np.ndarray:
    filter_size = int(np.mean(binary_image.shape) // 10)
    filter_size = filter_size + 1 if (filter_size % 2 == 0) else filter_size

    img = cv2.dilate(binary_image, (filter_size, filter_size), iterations=15)
    __debug_show_image(img)
    n, img = cv2.connectedComponents(img, connectivity=8, ltype=cv2.CV_16U)

    unique, count = np.unique(img, return_counts=True)
    # Find index of max element except the background (backgrounds is always 0)
    # Added 1 to compensate removing the background
    max = np.argmax(count[1:]) + 1

    max_elements = []
    for i in range(1, n):
        i_max = count[i] / count[max]
        if i_max >= 0.25:
            max_elements.append(unique[i])

    img[np.isin(img, max_elements)] = 1 << 15  # img is uint16

    binary_image[img != 1 << 15] = 0
    return binary_image


def __debug_show_image(img):
    if __DEBUG__:
        cv2.imshow("DEBUG", img)
        cv2.waitKey()


def __rotate_image(img: np.ndarray, angle_in_degrees) -> np.ndarray:
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)

    # Rotation Matrix:
    # [ cos -sin ]
    # [ sin  cos ]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_in_degrees, scale=1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # New dimensions of the image
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Add translation to rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height), flags=cv2.WARP_FILL_OUTLIERS,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def __get_line_length(line):
    x1, y1, x2, y2 = np.array(line.copy()).flatten()
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Get Line's slope angle in degrees
def __get_line_angle(line):
    x1, y1, x2, y2 = np.array(line.copy()).flatten()
    x = x2 - x1
    y = y2 - y1
    return math.degrees(math.atan2(y, x))


def __get_bounding_lines(hull_points) -> np.ndarray:
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)

    lines = [[hull_points[0], hull_points[1]]]
    old_angle = __get_line_angle(lines[0])
    for i in range(2, len(hull_points)):
        line = [hull_points[i - 1], hull_points[i]]
        new_angle = abs(__get_line_angle(line))
        if abs(old_angle - new_angle) <= 45:
            lines[-1][1] = hull_points[i]
        else:
            lines.append(line)
        old_angle = abs(__get_line_angle(lines[-1]))

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


# Return Top-Left , Top-Right , Bottom-Right , Bottom-Left boundary boundary points
# They are used for perspective fixing, therefore we use only left and right lines to get points
def __get_boundary_points(boundary_lines) -> np.ndarray:
    bounding_points = np.zeros((4, 2), dtype=np.float32)
    for i, line in enumerate(boundary_lines):
        next_line = boundary_lines[(i + 1) % len(boundary_lines)]
        intersection = __get_intersection(line, next_line)
        bounding_points[i] = intersection
    return bounding_points


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


# points is array in form of [[x1,y1] , [x2,y2] ...]
def __any_point_outside_image(img: np.ndarray, points):
    for point in points:
        if point[0] < 0 or point[0] >= img.shape[1]:
            return True
        if point[1] < 0 or point[1] >= img.shape[0]:
            return True
    return False


def get_perspective_transformation_matrix(img: np.ndarray) -> np.ndarray:
    all_points = cv2.findNonZero(img)
    hull_points: np.ndarray = cv2.convexHull(all_points)

    bounding_lines = __get_bounding_lines(hull_points)
    # debug_image = img.copy() // 4
    # cv2.drawContours(debug_image, np.int32(bounding_lines), -1, (160, 0, 0), 2)
    # __debug_show_image(debug_image)

    if bounding_lines.shape[0] < 4:
        return np.eye(3)

    try:
        bounding_points = __get_boundary_points(bounding_lines)
    except ArithmeticError:
        return np.eye(3)

    if __any_point_outside_image(img, bounding_points):
        return np.eye(3)

    # cv2.drawContours(debug_image, bounding_points.reshape((4, 1, 2)).astype(int), -1, (255, 0, 0), 10)
    # __debug_show_image(debug_image)

    x, y, w, h = cv2.boundingRect(bounding_points)
    rectangle_points = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    if are_points_same_skew(bounding_points, rectangle_points, img.shape):
        return np.eye(3)

    transformation_matrix = cv2.getPerspectiveTransform(bounding_points, rectangle_points)
    return transformation_matrix


def are_points_same_skew(bounding_points, rectangle_points, img_shape):
    epsilon_x = math.ceil(1 + img_shape[1] / 100)
    epsilon_y = math.ceil(1 + img_shape[0] / 100)

    x1, y1, x2, y2, x3, y3, x4, y4 = rectangle_points.flatten()
    X1, Y1, X2, Y2, X3, Y3, X4, Y4 = bounding_points.flatten()

    if (abs(x1 - X1) <= epsilon_x and abs(x2 - X2) <= epsilon_x
            and abs(x3 - X3) <= epsilon_x and abs(x4 - X4) <= epsilon_x):
        return True
    if (abs(y1 - Y1) <= epsilon_y and abs(y2 - Y2) <= epsilon_y
            and abs(y3 - Y3) <= epsilon_y and abs(y4 - Y4) <= epsilon_y):
        return True
    return False


def __crop_image(binary_image: np.ndarray):
    all_points = cv2.findNonZero(binary_image)
    x, y, w, h = cv2.boundingRect(all_points)
    border_x = w // 20
    border_y = h // 10
    left = max(0, x - border_x)
    right = min(binary_image.shape[1], x + w + border_x)
    top = max(0, y - border_y)
    bottom = min(binary_image.shape[0], y + h + border_y)
    return binary_image[top:bottom, left:right]


def __get_hough_lines(binary_image: np.ndarray):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = np.min(binary_image.shape) // 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = np.max(binary_image.shape) // 4
    max_line_gap = min_line_length / 10  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines


# Remove Extreme and Odd Values (Outliers) From The Array
def __reject_outliers(arr: np.ndarray) -> np.ndarray:
    return arr[abs(arr - np.mean(arr)) <= 1.2 * np.std(arr)]


def __get_rotation_angle_hough(binarized_image: np.ndarray):
    lines_endpoints = __get_hough_lines(binarized_image)

    data_type = [('length', float), ('angle', float)]
    lines_properties = np.zeros(len(lines_endpoints), dtype=data_type)

    for i, line in enumerate(lines_endpoints):
        lines_properties[i]['length'] = __get_line_length(line)
        lines_properties[i]['angle'] = __get_line_angle(line)

    lines_properties[::-1].sort(order='length')  # sort in descending order by length field

    return np.median(__reject_outliers(lines_properties[:10]['angle']))


def __get_rotation_angle(binary_image: np.ndarray):
    all_points = cv2.findNonZero(binary_image)
    center, (width, height), angle = cv2.minAreaRect(all_points)

    if width < height:
        angle += 90

    return angle
