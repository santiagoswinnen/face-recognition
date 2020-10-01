import cv2
from pca import pca_face_input, kpca_face_input

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def detect_faces(mode, eigenfaces, names):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(1)

    while True:
        ret, frame = video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            cmd = cv2.waitKey(50)
            if cmd == ord('r'):
                i = 0
                for (x, y, w, h) in faces:
                    new_h = int(1.1 * h)
                    h_padd = int(0.3 * h)

                    face_img = gray[y - h_padd:y + new_h, x:x + w]
                    face_img = cv2.resize(face_img, (92, 112))

                    print("[INFO] Object found. Saving locally.")
                    cv2.imwrite('cv_added_faces/faces_' + str(i) + '.jpg', face_img)
                    if mode == 'pca':
                        label = pca_face_input(eigenfaces, face_img)
                    else:
                        label = kpca_face_input(eigenfaces, face_img)
                    name = names[label]
                    if name is None:
                        name = 'Unknown'
                    cv2.putText(frame, name, (x, y - h_padd), font, fontScale, fontColor, lineType)
                    i += 1
                # Display the resulting frame
                cv2.imshow('Video', frame)
                cv2.waitKey(2000)
            elif cmd == ord('e'):
                break
            else:
                for (x, y, w, h) in faces:
                    new_h = int(1.1 * h)
                    h_padd = int(0.3 * h)
                    cv2.rectangle(frame, (x, y - h_padd), (x + w, y + new_h), (0, 255, 0), 2)
                    # Display the resulting frame
                    cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()
    return
