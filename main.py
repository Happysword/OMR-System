import cv2
import Binarization as binarization
from Dictionary import *
from commonfunctions import *
from skimage import exposure
from fix_orientation import fix_orientation
from staff import *
from segment import *
from NotesDetection import *
import io_utils
from features import extract_features
import pickle as pickle

args = io_utils.get_command_line_args()
__DEBUG__ = args.debug

# load the model from disk
loaded_model = pickle.load(open('Model.sav', 'rb'))

filenames = io_utils.get_filenames(args.input_path)
for filename in filenames:
    try:  # Use to ignore any error and go to next file instead of exiting

        originalImage = io_utils.read_grayscale_image(args.input_path, filename)

        # Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
        fixed_orientation = fix_orientation(originalImage, __DEBUG__)

        # Binarization Step
        binary_image = 255 * binarization.AdaptiveThresholding(fixed_orientation, 3)  # give good results

        show_images([fixed_orientation, binary_image])

        # io_utils.write_image(fixed_orientation, args.output_path, filename)

        segmented_staffs_array = segment_staff(binary_image)
        # show_images(segmented_staffs_array)

        # # # Getting Staff features
        staffs = []
        for segment in segmented_staffs_array:
            staffs.append(Staff(np.uint8(segment)))
        for i in staffs:
            show_images([i.lines, i.notes], ["Detected Lines", "Detected notes"])

        FinalOutput = "{\n"

        for staff in staffs:
            Symbols,borders = segment_symbols(staff.notes)
            show_images(Symbols)
            # print(segment_symbols(staff.notes))
            # print(staff.positions)
            noteObject = NotesPositions(staff.image, staff.positions, staff.space, staff.notes,staff.thickness)
            #Extract features and predict value
            staffObject = []
            for i,symbol in enumerate(Symbols):
                symbolObj = []
                features = extract_features(symbol, 'all') 
                value = loaded_model.predict([features])
                symbolObj.append( str(value[0]) )
                symbolObj.append(borders[i])
                staffObject.append(symbolObj)

            FinalOutput += TranslateStaff(staffObject,noteObject)
        
        FinalOutput = FinalOutput[:-2]
        FinalOutput += "\n}"
        FinalOutput = FixSpecialShapes(FinalOutput)
        print(FinalOutput)
        # for (i,symbol) in enumerate(symbols):
        #     io_utils.write_image(symbol,"NewDataSet",str(i)+'.png')

        text_file = open("Output.txt", "w")
        text_file.write(FinalOutput)
        text_file.close()

    except Exception as e:
        print(e)
        pass
