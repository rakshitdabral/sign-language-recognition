import cv2  #for capture
import os   #for file handling

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
        ret, frame = cap.read()
        cv2.putText(frame,'Press S', (100,50), cv2.FONT_HERSHEY_PLAIN, 1 , (255,0,0), 2, cv2.LINE_4)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(25)==ord('s')):
            break

    counter = 0
    while counter < dataSize:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dataDir, str(i), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
