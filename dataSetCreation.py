import os
import cv2
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# designing our hand using media pipe query
hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3)

dataDir = "./data"


data = []
label = []

for dir_ in os.listdir(dataDir): # use to locate folder inside data
    for imagePath in os.listdir(os.path.join(dataDir,dir_))[:1]: # locate image inside that directory
        xCor = []
        yCor= []
        dataAux= []

        img = cv2.imread(os.path.join(dataDir,dir_,imagePath))  # reading image from that specific directory

        imgConverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting image from bgr to rgb  before processing it

        results = hands.process(imgConverted) # processing hand

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    xCor.append(x)
                    yCor.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    dataAux.append(x - min(xCor))
                    dataAux.append(y - min(yCor))


            # mp_drawing.draw_landmarks(
            #     imgConverted,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())

            data.append(dataAux)
            label.append(dir_)

#             plt.figure()
#             plt.imshow(imgConverted)
# plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': label}, f)
f.close()