import cv2  #for capture
import os   #for file handling
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3)


dataDir  = "./data"


# Case handling if the path does not exist in our system
if not os.path.exists(dataDir): # we check if the path exist or not
    os.makedirs(dataDir) # if the dir does not exist we create one

dataClass= 20
dataSize= 300

cap = cv2.VideoCapture(0)

for i in range(dataClass):
    if not os.path.exists(os.path.join(dataDir,str(i))): # we check if we have path named  ./data/0 ./data/1 ./data/2
            os.makedirs(os.path.join(dataDir,str(i))) # if path does not exist we create a path

    done = False

    while True:
        ret, frame = cap.read() # setting up frame to capture user
        cv2.putText(frame,'Press S', (100,50), cv2.FONT_HERSHEY_PLAIN, 1 , (255,0,0), 2, cv2.LINE_4) # setting up frame style

        frame.flags.writeable  = False
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('frame',cv2.flip(image,1)) # showing frame

        if(cv2.waitKey(25)==ord('s')): # displays window for a given time frame
            break

    counter = 0
    while counter < dataSize:
        ret, frame = cap.read() # reading from frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dataDir, str(i), '{}.jpg'.format(counter)), frame) # saving read image into folder

        counter += 1

cap.release()
cv2.destroyAllWindows()
