import argparse
import cv2
import time
import os
import imutils
from imutils.video import VideoStream
import numpy as np


def start():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--haarcas", required=True, help="Provide path to the harcascade XML file for face detection")
    ap.add_argument("-o", "--out", required=True, help="Path to output pictures")
    ap.add_argument("-pc", "--picamera", required=False,
                    help="Use PiCamera class for video instead of Opencv VideoStream")
    args = vars(ap.parse_args())
    haarcascade_xml = args["haarcas"]
    photos_folder_path = args["out"]

    face = cv2.CascadeClassifier(haarcascade_xml)

    vs = VideoStream(src=0, framerate=20).start()

    # for camera to warm-up
    time.sleep(1)
    total = 0

    while True:
        frame = vs.read()
        original_frame = frame.copy()

        frame = imutils.resize(frame, width=400)

        face_rect = face.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30))
        cv2.imshow("Frame", frame)
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("k"):
                # capture picture
                p = os.path.sep.join([photos_folder_path, f"{str(total).zfill(5)}.png"])
                print(f"Path - {p}")
                cv2.imwrite(p, original_frame)
                total += 1

    cv2.destroyAllWindows()
    vs.stop()


def face_date_set_using_pre_trained_model():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="Path to output pictures")
    ap.add_argument("-pc", "--picamera", required=False,
                    help="Use PiCamera class for video instead of Opencv VideoStream")
    ap.add_argument("-fn", "--fileno", required=True,
                    help="File number sequence ")
    args = vars(ap.parse_args())
    photos_folder_path = args["out"]
    file_sequence_number_start = args['fileno']
    print(cv2.__version__)

    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(0)

    total = int(file_sequence_number_start)
    while cap.isOpened():
        res, image = cap.read()
        original_frame = image.copy()
        width_we_need = 600
        # image = cv2.imread("../Data/Images/face4.jpg")
        (h, w) = image.shape[:2]

        r = width_we_need / float(w)
        dim = (width_we_need, int(h * r))

        image = cv2.resize(src=image, dsize=dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Reshaped", image)

        (h, w) = image.shape[:2]
        print(h, w)

        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300))
                                           , 1.0, (300, 300),
                                           (104.0, 177.0, 123.0),
                                           swapRB=False, crop=False)
        import time

        t = time.time()
        print(t)
        detector.setInput(image_blob)
        detections = detector.forward()
        tt = time.time() - t
        print(tt)
        face = None
        if len(detections) > 0:
            print(f"Detections shape: {detections.shape}")
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fh, fw) = face.shape[:2]
                print(f"face hei, wid: {fh}, {fw}")

                cv2.rectangle(image, (startX, startY), (endX, endY), [255, 0, 0], 10)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("k"):
                    # capture picture
                    p = os.path.sep.join([photos_folder_path, f"{str(total).zfill(5)}.jpg"])
                    print(f"Path - {p}")
                    cv2.imwrite(p, original_frame)
                    total += 1

        cv2.imshow("Face box", image)
        # cv2.waitKey(0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_date_set_using_pre_trained_model()
