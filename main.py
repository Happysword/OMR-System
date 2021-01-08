import cv2
import Binarization as binarization
from Dictionary import *
from debug_utils import *
from skimage import exposure
from skimage import transform
from fix_orientation import fix_orientation
from staff import *
from segment import *
from NotesDetection import *
import io_utils
from features import extract_features
import pickle as pickle
import traceback

args = io_utils.get_command_line_args()

# load the model from disk
loaded_model = pickle.load(open('Model.sav', 'rb'))

filenames = io_utils.get_filenames(args.input_path)
for filename in filenames:
    try:  # Use to ignore any error and go to next file instead of exiting

        originalImage = io_utils.read_grayscale_image(args.input_path, filename)

        # Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
        fixed_orientation = fix_orientation(originalImage)

        # Binarization Step
        binary_image = 255 * binarization.AdaptiveThresholding(fixed_orientation, 3)  # give good results

        debug_show_images([fixed_orientation, binary_image])

        # io_utils.write_image(fixed_orientation, args.output_path, filename)
        segmented_staffs_array = segment_staff(binary_image)
        # debug_show_images(segmented_staffs_array)

        # # # Getting Staff features
        staffs = []
        for segment in segmented_staffs_array:
            staffs.append(Staff(np.uint8(segment)))
        for i in staffs:
            debug_show_images([i.lines, i.notes], ["Detected Lines", "Detected notes"])

        if len(staffs) > 1:
            io_utils.write_line_file("{", args.output_path, filename)

        clefFound = False
        rotateAgain = False
        for staff_number, staff in enumerate(staffs):
            Symbols, borders = segment_symbols(staff.notes)
            debug_show_images(Symbols)
            # debug_print(segment_symbols(staff.notes))
            # debug_print(staff.positions)
            noteObject = NotesPositions(staff.image, staff.positions, staff.space, staff.notes, staff.thickness)
            # Extract features and predict value
            staffObject = []
            for i, symbol in enumerate(Symbols):
                symbolObj = []
                features = extract_features(symbol, 'all')
                value = loaded_model.predict([features])
                if value == 'clef':
                    clefFound = True
                if clefFound == False and i > (len(Symbols) // 4) + 1:
                    rotateAgain = True
                    break
                symbolObj.append(str(value[0]))
                symbolObj.append(borders[i])
                staffObject.append(symbolObj)

            # Try again
            if rotateAgain:
                break

            FinalOutput = TranslateStaff(staffObject, noteObject)
            if staff_number < (len(staffs) - 1):
                FinalOutput += " ,"
            io_utils.write_line_file(FixSpecialShapes(FinalOutput), args.output_path, filename)

        ###############################
        # Rotate 180 then try again
        ###############################
        if rotateAgain:
            # print(type(binary_image[0][0]))
            binary_image = transform.rotate(binary_image, 180)
            binary_image = 255 * binarization.AdaptiveThresholding(binary_image, 3)

            debug_show_images([binary_image])

            segmented_staffs_array = segment_staff(binary_image)

            # # # Getting Staff features
            staffs = []
            for segment in segmented_staffs_array:
                staffs.append(Staff(np.uint8(segment)))
            for i in staffs:
                debug_show_images([i.lines, i.notes], ["Detected Lines", "Detected notes"])

            if len(staffs) > 1:
                io_utils.write_line_file("{", args.output_path, filename)

            for staff_number, staff in enumerate(staffs):
                Symbols, borders = segment_symbols(staff.notes)
                debug_show_images(Symbols)
                # debug_print(segment_symbols(staff.notes))
                # debug_print(staff.positions)
                noteObject = NotesPositions(staff.image, staff.positions, staff.space, staff.notes, staff.thickness)
                # Extract features and predict value
                staffObject = []
                for i, symbol in enumerate(Symbols):
                    symbolObj = []
                    features = extract_features(symbol, 'all')
                    value = loaded_model.predict([features])
                    symbolObj.append(str(value[0]))
                    symbolObj.append(borders[i])
                    staffObject.append(symbolObj)

                FinalOutput = TranslateStaff(staffObject, noteObject)
                if staff_number < (len(staffs) - 1):
                    FinalOutput += " ,"
                io_utils.write_line_file(FixSpecialShapes(FinalOutput), args.output_path, filename)

        if len(staffs) > 1:
            io_utils.write_file("}", args.output_path, filename)

        # for (i,symbol) in enumerate(symbols):
        #     io_utils.write_image(symbol,"NewDataSet",str(i)+'.png')

    except:
        print("____________________________________________________________")
        print("[Error]: Occurred in file", filename)
        print(traceback.format_exc())
        print("____________________________________________________________")
