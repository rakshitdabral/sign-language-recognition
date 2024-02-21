from keras.models import model_from_json
import cv2
import numpy as np

modelJson = open('modelVersion2.0.1.json', 'r').read()

model = model_from_json(modelJson)
model.load_weights('modelVersion2.0.1.h5')

def features(image):
    feature = np.array(image)
    feature = feature.reshape(1,image.shape[0],image.shape[1],3)
    return feature / 255.0

cap = cv2.VideoCapture(0)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.resize(cropframe, (64,64))
    cropframe = features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = labels[pred.argmax()]
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("output", frame)
    cv2.waitKey(27)

cap.release()
cv2.destroyAllWindows()