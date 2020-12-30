import cv2
import Binarization as binarization
from commonfunctions import *
from skimage import exposure
from fix_orientation import fix_orientation
from staff import *
from segment import *
from NotesDetection import *
import io_utils

args = io_utils.get_command_line_args()
__DEBUG__ = args.debug

for filename in io_utils.get_filenames(args.input_path):
    try:  # Use to ignore any error and go to next file instead of exiting

        originalImage = io_utils.read_grayscale_image(args.input_path, filename)

        # Binarization Step
        thre4 = binarization.AdaptiveThresholding(originalImage, 3, 21)  # give good results

        # Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
        fixed_orientation = fix_orientation(thre4, __DEBUG__)
        # show_images([fixed_orientation])

        # utils.write_image(fixed_orientation, args.output_path, filename)

        segmented_staffs_array = segment_staff(thre4 * 255)

        # Getting Staff features

        staffs = []
        for segment in segmented_staffs_array:
            staffs.append(Staff(np.uint8(segment)))
        for i in staffs:
            show_images([i.lines, i.notes], ["Detected Lines", "Detected notes"])

        symbols = []
        for staff in staffs:
            temp = segment_symbols(staff.notes)
            symbols = symbols + temp
            show_images(temp)
            # print(segment_symbols(staff.notes))
            # print(staff.positions)
            # notePoints,notesNames = NotesPositions(thre4,staff.positions,staff.space)
            # print(notesNames)
    except Exception as e:
        print(e)
        pass
