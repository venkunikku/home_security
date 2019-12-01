import argparse
import cv2
import time
import os
import imutils
from imutils.video import VideoStream



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


if __name__ == "__main__":
    start()
