import cv2 # cv for computer vision
import os # os for file handling

# defining the directory to store dataset
directory = 'data/'
# print(os.getcwd())

# check if the directory or the folder exist if it does we do nothing
# else we will create a working directory with the name
if not os.path.exists(directory):
    os.mkdir(directory)

# nothing directory to hold empty photos
if not os.path.exists(f'{directory}/nothing'):
    os.mkdir(f'{directory}/nothing')

# del to hold del sign
if not os.path.exists(f'{directory}/del'):
    os.mkdir(f'{directory}/del')

# space to hold space sign
if not os.path.exists(f'{directory}/space'):
    os.mkdir(f'{directory}/space')


# printing ascii character from A to Z
for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')


#defining camera from open cv
cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read() # reading camera
    count = {
        'a': len(os.listdir(directory + "/A")),
        'b': len(os.listdir(directory + "/B")),
        'c': len(os.listdir(directory + "/C")),
        'd': len(os.listdir(directory + "/D")),
        'e': len(os.listdir(directory + "/E")),
        'f': len(os.listdir(directory + "/F")),
        'g': len(os.listdir(directory + "/G")),
        'h': len(os.listdir(directory + "/H")),
        'i': len(os.listdir(directory + "/I")),
        'j': len(os.listdir(directory + "/J")),
        'k': len(os.listdir(directory + "/K")),
        'l': len(os.listdir(directory + "/L")),
        'm': len(os.listdir(directory + "/M")),
        'n': len(os.listdir(directory + "/N")),
        'o': len(os.listdir(directory + "/O")),
        'p': len(os.listdir(directory + "/P")),
        'q': len(os.listdir(directory + "/Q")),
        'r': len(os.listdir(directory + "/R")),
        's': len(os.listdir(directory + "/S")),
        't': len(os.listdir(directory + "/T")),
        'u': len(os.listdir(directory + "/U")),
        'v': len(os.listdir(directory + "/V")),
        'w': len(os.listdir(directory + "/W")),
        'x': len(os.listdir(directory + "/X")),
        'y': len(os.listdir(directory + "/Y")),
        'z': len(os.listdir(directory + "/Z")),
        'nothing': len(os.listdir(directory + "/nothing")),
        'space': len(os.listdir(directory + "/space")),
        'del': len(os.listdir(directory + "/del")),
    }

    row = frame.shape[1]
    col = frame.shape[0]

    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2) # drawing a rectangle over the frame
    cv2.imshow("data", frame) # showing the main frame

    frame = frame[40:300, 0:300] # drawing a frame with specific width and height  to crop out the excess image
    cv2.imshow("ROI", frame)

    # converting colour from bgr to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (200, 200))

    #  if interrupt & 0xFF == ord('A') checks if a value stored in the variable interrupt corresponds to the letter 'A' pressed on the keyboard
    # saving the data into the secific folder and name with alphabet and its number
    interrupt = cv2.waitKey(10)


    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(os.path.join(directory + 'A/' + str(count['a'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(os.path.join(directory + 'B/' + str(count['b'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(directory + 'C/' + str(count['c'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(os.path.join(directory + 'D/' + str(count['d'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(os.path.join(directory + 'E/' + str(count['e'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(os.path.join(directory + 'F/' + str(count['f'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(os.path.join(directory + 'G/' + str(count['g'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(os.path.join(directory + 'H/' + str(count['h'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(os.path.join(directory + 'I/' + str(count['i'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(os.path.join(directory + 'J/' + str(count['j'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(os.path.join(directory + 'K/' + str(count['k'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(os.path.join(directory + 'L/' + str(count['l'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(os.path.join(directory + 'M/' + str(count['m'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(os.path.join(directory + 'N/' + str(count['n'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(os.path.join(directory + 'O/' + str(count['o'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(os.path.join(directory + 'P/' + str(count['p'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(os.path.join(directory + 'Q/' + str(count['q'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(os.path.join(directory + 'R/' + str(count['r'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(os.path.join(directory + 'S/' + str(count['s'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(os.path.join(directory + 'T/' + str(count['t'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(os.path.join(directory + 'U/' + str(count['u'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(os.path.join(directory + 'V/' + str(count['v'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(os.path.join(directory + 'W/' + str(count['w'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(os.path.join(directory + 'X/' + str(count['x'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(os.path.join(directory + 'Y/' + str(count['y'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(os.path.join(directory + 'Z/' + str(count['z'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory + 'nothing/' + str(count['nothing'])) + '.jpg', frame)
    if interrupt & 0xFF == ord('/'):
        cv2.imwrite(os.path.join(directory + 'space/' + str(count['space'])) + '.jpg', frame)
    if interrupt & 0xFF == ord(','):
        cv2.imwrite(os.path.join(directory + 'del/' + str(count['del'])) + '.jpg', frame)


