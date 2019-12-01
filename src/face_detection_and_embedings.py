import cv2
import imutils
import numpy as np
import argparse
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import VideoStream
import time


def detect_face_from_input_images_and_save_to_pickle():
    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="Path to output pictures")

    args = vars(ap.parse_args())
    photos_folder_path = args["out"]

    our_embeddings = []
    our_names = []
    total = 0

    for file_path in get_files(photos_folder_path):
        name = file_path.split(os.path.sep)[-2]
        print(file_path, name)

        image = cv2.imread(file_path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                           swapRB=False, crop=False)
        detector.setInput(image_blob)
        detections = detector.forward()

        if len(detections) > 0:
            print("Detections are >0", detections.shape,
                  detections[0, 0, 0, 3:7].shape)  # index 3 value is varying (1, 1, 144, 7)

            i = np.argmax(detections[0, 0, :, 2])
            print(f"i argmax value: {i}")

            confidenc = detections[0, 0, i, 2]

            if confidenc > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startx, starty, endx, endy) = box.astype('int')

                face = image[starty:endy, startx:endx]
                (fh, fw) = face.shape[:2]

                if fw < 30 or fh < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                print(f"Vector Shape: {vec.shape}")

                our_names.append(name)
                our_embeddings.append(vec.flatten())
                total += 1
    print(f"Total: {total}")
    data = {"embeddings": our_embeddings, "names": our_names}
    print(data)
    # with open("../Data/pickle_saving/embeddings.pickle", "wb") as f:
    #     print("Saving the embeddings as pickle")
    #     f.write(pickle.dumps(data))
    #     f.close()
    fi = open("../Data/pickle_saving/embeddings.pickle", "wb")
    fi.write(pickle.dumps(data))
    fi.close()


def train_face_using_ml(path_to_embeddings=None, path_to_save_recognizer="../Data/pickle_saving/recognizer.pickle",
                        path_to_save_labels="../Data/pickle_saving/le.pickle"):
    data = None
    with open(path_to_embeddings, 'rb') as f:
        data = pickle.loads(f.read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    print(labels)

    recognizer = SVC(kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(path_to_save_recognizer, "wb") as fi:
        fi.write(pickle.dumps(recognizer))

    with open(path_to_save_labels, "wb") as lab:
        lab.write(pickle.dumps(le))


def get_files(path_to_folder=None):
    for root_dir, dir_name, file_name in os.walk(path_to_folder):
        # print(f"Root_dir: {root_dir}, dirName: {dir_name}, FilesName:{file_name}")
        for f in file_name:
            # print(f"Root_dir: {root_dir}, dirName: {dir_name}, FilesName:{f}")
            p = os.path.join(root_dir, f)
            yield p


def recognize_faces():
    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    recognizer = None
    le = None
    with open("../Data/pickle_saving/recognizer.pickle", "rb") as rec:
        recognizer = pickle.loads(rec.read())

    with open("../Data/pickle_saving/le.pickle", "rb") as lab_encod:
        le = pickle.loads(lab_encod.read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                           swapRB=False,
                                           crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startx, starty, endx, endy) = box.astype("int")

                face = frame[starty:endy, startx:endx]
                (fh, fw) = face.shape[:2]

                if fw < 20 or fh < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = starty - 10 if starty - 10 > 10 else starty + 10

                cv2.rectangle(frame, (startx - 10, y), (startx + 50, y + 10),
                              (30, 90, 120), -1)

                cv2.rectangle(frame, (startx, starty), (endx, endy),
                              (0, 255, 0), 2)
                cv2.putText(frame, text, (startx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

    vs.stop()
    cv2.destroyAllWindows()


def recognize_faces_using_ncs():
    #detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
    #                                    "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")


    #detector = cv2.dnn.readNet("../caffe_models/ncs2_with_mean/res10_300x300_ssd_iter_140000.xml",
    #                           "../caffe_models/ncs2_with_mean/res10_300x300_ssd_iter_140000.bin")

    detector = cv2.dnn.readNet("../caffe_models/ncs2/res10_300x300_ssd_iter_140000.xml",
                               "../caffe_models/ncs2/res10_300x300_ssd_iter_140000.bin")

    detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    recognizer = None
    le = None
    with open("../Data/pickle_saving/recognizer.pickle", "rb") as rec:
        recognizer = pickle.loads(rec.read())

    with open("../Data/pickle_saving/le.pickle", "rb") as lab_encod:
        le = pickle.loads(lab_encod.read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                           swapRB=False,
                                           crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startx, starty, endx, endy) = box.astype("int")

                face = frame[starty:endy, startx:endx]
                (fh, fw) = face.shape[:2]

                if fw < 20 or fh < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = starty - 10 if starty - 10 > 10 else starty + 10

                cv2.rectangle(frame, (startx - 10, y), (startx + 50, y + 10),
                              (30, 90, 120), -1)

                cv2.rectangle(frame, (startx, starty), (endx, endy),
                              (0, 255, 0), 2)
                cv2.putText(frame, text, (startx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

    vs.stop()
    cv2.destroyAllWindows()


def check_cv2():
    print(cv2.__version__)

    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        res, image = cap.read()

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

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                  (96, 96),
                                                  (0, 0, 0),
                                                  swapRB=False, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()
                print(f"Vec shape: {vec.shape}")
                # print(f"Vec:{vec}")

        # cv2.imshow("Face", face)
        cv2.imshow("Face box", image)
        # cv2.waitKey(0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_cv2()

    # detect_face_from_input_images_and_save_to_pickle()
    #
    # train_face_using_ml(path_to_embeddings="../Data/pickle_saving/embeddings.pickle")

    # recognize_faces()

    recognize_faces_using_ncs()
