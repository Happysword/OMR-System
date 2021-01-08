shapesNames = {
    "clef": "",
    "a_4": "/4",
    "a_1": "/1",
    "a_2": "/2",
    "a_8": "/8",
    "b_8": "/8",
    "a_16": "/16",
    "b_16": "/16",
    "a_32": "/32",
    "b_32": "/32",
    "sharp": "#",
    "natural": "",
    "flat": "&",
    "double_sharp": "##",
    "double_flat": "&&",
    "dot": ".",
    "barline": "",
    "chord": "",
    "t_2_2": "\meter<\"2/2\">",
    "t_2_4": "\meter<\"2/4\">",
    "t_3_4": "\meter<\"3/4\">",
    "t_3_8": "\meter<\"3/8\">",
    "t_4_4": "\meter<\"4/4\">",
    "t_6_8": "\meter<\"6/8\">",
    "t_9_8": "\meter<\"9/8\">",
    "t_12_8": "\meter<\"12/8\">"
}

notesWithHeads = ["a_4", "a_1", "a_2", "a_8", "b_8", "a_16", "b_16", "a_32", "b_32"]
specialShapes = ["#", "##", "&&", "&"]
meters = ["\meter<\"2/2\">", "\meter<\"2/4\">", "\meter<\"3/4\">", "\meter<\"3/8\">", "\meter<\"4/4\">",
          "\meter<\"6/8\">", "\meter<\"9/8\">", "\meter<\"12/8\">"]


# noteObject = [X-pos , The note Name , IsHollow ]
# shapeObject = [ The shape label , (X_min,X_max) ]
def TranslateStaff(shapeObject, noteObject):
    FinalOutput = "[ "
    for shape in shapeObject:

        # TODO : split notesWithHeads into solid and hollow to avoid taking wrong points
        # TODO : if not found a point make random postiion
        if shape[0] in notesWithHeads:
            x_min = shape[1][0]
            x_max = shape[1][1]

            for note in noteObject:
                if note[0] >= x_min and note[0] <= x_max:
                    FinalOutput += note[1] + shapesNames[shape[0]] + " "

        elif shape[0] == "dot":
            FinalOutput += shapesNames[shape[0]]

        elif shape[0] == "chord":
            x_min = shape[1][0]
            x_max = shape[1][1]

            chordArr = []
            for note in noteObject:
                if x_min <= note[0] <= x_max:
                    chordArr.append(note[1])

            chordArr = sorted(chordArr)
            
            FinalOutput += "{"
            for chord in chordArr:
                FinalOutput += chord + ","

            FinalOutput = FinalOutput[:-1]
            FinalOutput += "} "

        else:
            FinalOutput += shapesNames[shape[0]] + " "

    FinalOutput += "]"

    return FinalOutput


def FixSpecialShapes(outputString):
    words = outputString.strip().split(' ')

    for i, word in enumerate(words):
        if word in specialShapes:
            if i == len(words) - 1:
                break
            else:
                if "/" in words[i + 1] and words[i + 1] not in meters:
                    index = 1
                    out = words[i + 1][:index] + word + words[i + 1][index:]
                    words.remove(words[i + 1])
                    words.insert(i + 1, out)

    modifiedOutputString = ""
    for word in words:
        if word not in specialShapes and word != '':
            modifiedOutputString += word + " "
    return modifiedOutputString
