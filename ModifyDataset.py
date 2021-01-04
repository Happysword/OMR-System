import Binarization as binarization
import cv2
import io_utils
import numpy as np

args = io_utils.get_command_line_args()
__DEBUG__ = args.debug

def __get_cropping_rectangle(binary_image: np.ndarray):
    all_points = cv2.findNonZero(binary_image)
    x, y, w, h = cv2.boundingRect(all_points)
    border_x = w // 20
    border_y = h // 10
    left = max(0, x - border_x)
    right = min(binary_image.shape[1], x + w + border_x)
    top = max(0, y - border_y)
    bottom = min(binary_image.shape[0], y + h + border_y)
    return top, bottom, left, right

for filename in io_utils.get_filenames(args.input_path):
    try:  # Use to ignore any error and go to next file instead of exiting

        originalImage = io_utils.read_grayscale_image(args.input_path, filename)

        # Binarization Step
        binary_image = binarization.GlobalThresholding(originalImage)

        binary_image = 255 - binary_image

        top, bottom, left, right = __get_cropping_rectangle(binary_image)
        binary_image = binary_image[top:bottom, left:right]


        io_utils.write_image(binary_image, args.output_path, filename)
            
    except Exception as e:
        print(e)
        pass

