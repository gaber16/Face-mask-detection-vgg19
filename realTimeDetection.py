import tensorflow as tf
import cv2
import numpy as np

haar_cascade = cv2.CascadeClassifier('path for haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('Path for your model')

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret:
        print('Nothing captured using front camera!')
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(image, 1.2 , 10)



    for (x,y,w,h) in faces_rect:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (150,150))

        predictions = model.predict(np.expand_dims(face, axis=0), verbose=0)
        mask_label = 'Mask on' if predictions[0][0]>=0.9 else "No mask"
        if mask_label == 'Mask on':
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, mask_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, mask_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Mask detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()