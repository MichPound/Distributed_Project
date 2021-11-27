# Code References:
    # https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask.
    # https://pylessons.com/YOLOv3-WebCam/.
    # https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/.
    # https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html.
    # https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python.

# Project imports.
from flask import Flask, render_template, Response
import cv2
import pyautogui
import numpy as np
import time
import mediapipe as mp

app = Flask(__name__)

# Grab camera feed.
camera = cv2.VideoCapture(0)

def prep_frames():
    start_time = time.time()
    display_time = 2
    fps = 0

    # Loading the YOLO network.
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # Preparing labels.
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]

    while True:
        success, frame = camera.read()
        # The fllowing three lines can be used for screen capture if no webcam is available.
        # frame = pyautogui.screenshot()
        # frame = np.array(frame)
        # success = True

        if not success:
            break
        else:
            height, width, channels = frame.shape

            # Making input blob fr YOLO.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(ln)

            # For each object detectied if over 50% confidence builds list of boxes.
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Adds boxes to frame using rectangles.
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            # Simple frame per second counter.
            fps += 1
            checkTime = time.time() - start_time
            if checkTime > display_time:
                fps = 0
                start_time = time.time()
            # Adds current fps to frame.
            cv2.putText(frame, str(int(fps / checkTime)), (10,55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 185, 118), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

# This allows for easy retrieval of the video frames.
@app.route('/video_feed')
def video_feed():
    return Response(prep_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Renders the basic html page.
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(app.run(host="0.0.0.0"), debug=True)
