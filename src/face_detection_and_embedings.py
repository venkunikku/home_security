import cv2
import imutils
import numpy as np
import argparse
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold, cross_val_score
from imutils.video import VideoStream
import time
from datetime import datetime
import matplotlib.pyplot as plt


def detect_face_from_input_images_and_save_to_pickle():
    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    args = processe_args()

    photos_folder_path = args["train_data"]

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


def test_a_model(model, create_embeddings=False, compare_models=False):
    print(f"Model we will use to evaluate the test results {model}")
    args = processe_args()

    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    photos_folder_path = args["test_data_path"]

    if create_embeddings:

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
        fi = open("../Data/pickle_saving_test/embeddings_test.pickle", "wb")
        fi.write(pickle.dumps(data))
        fi.close()

        # saving labels
        le = LabelEncoder()
        le.fit_transform(data["names"])
        with open("../Data/pickle_saving_test/le_test.pickle", "wb") as lab:
            lab.write(pickle.dumps(le))

    recognizer = None
    test_le = None
    test_embeddings = None
    with open(f"../Data/pickle_saving/recognizer_{model}.pickle", "rb") as rec:
        recognizer = pickle.loads(rec.read())

    with open("../Data/pickle_saving_test/le_test.pickle", "rb") as lab_encod:
        test_le = pickle.loads(lab_encod.read())

    with open("../Data/pickle_saving_test/embeddings_test.pickle", "rb") as test_emb:
        test_embeddings = pickle.loads(test_emb.read())

    print(test_embeddings["names"])
    predictions = recognizer.predict(test_embeddings["embeddings"])

    encoding = LabelEncoder()
    labels = encoding.fit_transform(test_embeddings["names"])
    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions))
    # print(roc_curve(labels, predictions))

    with open("../Data/pickle_saving/embeddings.pickle", "rb") as train_embeddings:
        train_embed = pickle.loads(train_embeddings.read())

    for file_path in get_files(photos_folder_path):
        name = file_path.split(os.path.sep)[-2]
        image_name = file_path.split(os.path.sep)[-1]
        #print(file_path, name)

        image = cv2.imread(file_path)
        frame = identify_person(detector, embedder, image, encoding, recognizer)
        cv2.putText(frame, f"Model: {model}", (10, 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0, 0, 0), 1)
        file_path = f"../Data/test_images_annotated/{model}/{name}_{image_name}"
        cv2.imwrite(file_path, frame)

    if compare_models:
        models = []
        models.append(('LDA', LinearDiscriminantAnalysis(n_components=3)))
        models.append(('SVC', SVC(kernel="linear", gamma='scale', probability=True)))
        models.append(('RF', RandomForestClassifier(n_estimators=1000)))
        models.append(('SGD', SGDClassifier(max_iter=1000, tol=1e-3, loss='log')))
        models.append(('GBC', GradientBoostingClassifier(n_estimators=1000)))
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = KFold(n_splits=5, random_state=24)
            cv_results = cross_val_score(model, train_embed["embeddings"], train_embed["names"],
                                         cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        fig = plt.figure()
        fig.suptitle('Comparison of the Models')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
def train_face_using_ml(path_to_embeddings=None, path_to_save_recognizer="../Data/pickle_saving/recognizer_{}.pickle",
                        path_to_save_labels="../Data/pickle_saving/le.pickle"):
    args = processe_args()
    data = None
    with open(path_to_embeddings, 'rb') as f:
        data = pickle.loads(f.read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    print(labels)
    model = args["ml_model"]
    if model == "SGD":
        recognizer = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
        recognizer.fit(data["embeddings"], labels)
    elif model == "SGD_mod_huber":
        recognizer = SGDClassifier(max_iter=1000, tol=1e-3, loss='modified_huber')
        recognizer.fit(data["embeddings"], labels)
    elif model == 'LDA':
        recognizer = LinearDiscriminantAnalysis(n_components=3)
        recognizer.fit(data["embeddings"], labels)
    elif model == "GBC":
        recognizer = GradientBoostingClassifier(n_estimators=1000)
        recognizer.fit(data["embeddings"], labels)
    elif model == 'RF':
        recognizer = RandomForestClassifier(n_estimators=1000)
        recognizer.fit(data["embeddings"], labels)
    else:
        model = 'SVC'
        recognizer = SVC(kernel="linear", gamma='scale', probability=True, decision_function_shape='ovo')
        recognizer.fit(data["embeddings"], labels)

    with open(path_to_save_recognizer.format(model), "wb") as fi:
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


def recognize_faces(model):
    args = processe_args()
    use_camera = args["cam"]

    detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
                                        "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    embedder = cv2.dnn.readNetFromTorch("../caffe_models/openface_nn4.small2.v1.t7")

    recognizer = None
    le = None
    with open(f"../Data/pickle_saving/recognizer_{model}.pickle", "rb") as rec:
        recognizer = pickle.loads(rec.read())

    with open("../Data/pickle_saving/le.pickle", "rb") as lab_encod:
        le = pickle.loads(lab_encod.read())

    if use_camera == "True" or use_camera is None:
        vs = VideoStream(src=0).start()
    else:
        print(args)
        video_file_path = args["video_file_path"]
        vs = cv2.VideoCapture(video_file_path)  # VideoStream(video_file_path).start()

    record = args["record"]
    out = None
    if record == "True":
        print("Recording")
        w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"width and height: {w},{h}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # *"XVID"
        out = cv2.VideoWriter(f"../Data/video/face_classified/video_{datetime.now}.mp4", fourcc, 20.0,
                              (600, 330))  # 20 frams

    time.sleep(2.0)

    try:
        while True:

            if use_camera == "True":
                frame = vs.read()
            else:
                time.sleep(.01)
                ret, frame = vs.read()
            frame = identify_person(detector, embedder, frame, le, recognizer)
            cv2.putText(frame, f"Model: {model}", (10, 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0, 0, 0), 1)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            if record == "True":
                print("Writing to the frame")
                out.write(frame)
            cv2.imshow("Frame", frame)
        cv2.destroyAllWindows()
        if use_camera == "True":
            vs.stop()
    except Exception as e:
        print("Exception:", e)
    finally:

        cv2.destroyAllWindows()
        if use_camera == "False":
            vs.release()
    if record == "True":
        print("Releasing out video stream")
        out.release()


def identify_person(detector, embedder, frame, le, recognizer):
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                       swapRB=False,
                                       crop=False)
    detector.setInput(image_blob)
    detections = detector.forward()
    # cv2.imshow("Frame", frame)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")

            face = frame[starty:endy, startx:endx]
            (fh, fw) = face.shape[:2]

            if fw < 20 or fh < 20:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = starty - 10 if starty - 10 > 10 else starty + 10

            cv2.rectangle(frame, (startx, y - 20), (startx + 135, y + 5),
                          (50, 205, 50), -1)

            cv2.rectangle(frame, (startx, starty), (endx, endy),
                          (50, 205, 50), 2)
            cv2.putText(frame, text, (startx, y),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 255, 255), 1)

            if name == "unknown":
                cv2.rectangle(frame, (startx, y - 20), (startx + 135, y + 5),
                              (0, 0, 225), -1)

                cv2.rectangle(frame, (startx, starty), (endx, endy),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startx, y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 255, 255), 1)

            # cv2.imshow("Frame", frame)
    #print(frame.shape)
    return frame


def recognize_faces_using_ncs():
    # detector = cv2.dnn.readNetFromCaffe("../caffe_models/deploy.prototxt",
    #                                    "../caffe_models/res10_300x300_ssd_iter_140000.caffemodel")

    # detector = cv2.dnn.readNet("../caffe_models/ncs2_with_mean/res10_300x300_ssd_iter_140000.xml",
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

                cv2.rectangle(frame, (startx, y - 20), (startx + 135, y + 5),
                              (50, 205, 50), -1)

                cv2.rectangle(frame, (startx, starty), (endx, endy),
                              (50, 205, 50), 2)
                cv2.putText(frame, text, (startx, y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 255, 255), 1)

                cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            # cv2.dnn.writeTextGraph(detector, "../Data/misc")
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

            if confidence > 0.8:
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


def processe_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cam", "--cam", required=False, help="Should camera be used")
    ap.add_argument("-vfp", "--video_file_path", required=False, help="Path of the video file to be processed")

    # Path to the pictures for training
    ap.add_argument("-td", "--train_data", required=False, help="Path to output pictures")

    # if we have to create the facial embeddings
    ap.add_argument("-embed", "--create_embeddings", required=False, help="create the facial embeddings")

    # Train the ML model to our faces
    ap.add_argument("-train_model", "--train_model", required=False, help="Train the model")

    # start recognizing face
    ap.add_argument("-face", "--face_recognize", required=False, help="Recognize faceß")

    ap.add_argument("-ml", "--ml_model", required=False, help="Recognize face")

    ap.add_argument("-r", "--record", required=False, help="Record the face")

    # Testing models
    ap.add_argument("-test_data", "--test_data_path", required=False, help="Testing data Path")
    ap.add_argument("-test", "--test", required=False, help="Testing the model")

    args = vars(ap.parse_args())

    return args


if __name__ == '__main__':
    program_arguments = processe_args()
    # check_cv2()

    if program_arguments["create_embeddings"] == "True":
        print("Creating Embeddings")
        detect_face_from_input_images_and_save_to_pickle()
    if program_arguments["train_model"] == 'True':
        train_face_using_ml(path_to_embeddings="../Data/pickle_saving/embeddings.pickle")

    if program_arguments["test"] == "True":
        model = program_arguments["ml_model"]
        test_a_model(model, create_embeddings=True, compare_models=True)

    if program_arguments["face_recognize"] == "True":
        model = program_arguments["ml_model"]
        recognize_faces(model)

    # recognize_faces_using_ncs()

    print(datetime.now)
