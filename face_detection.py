# Code References:
    # https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask.
    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html.

# Project imports.
from flask import Flask, render_template, Response
import cv2
import pyautogui
import time

app = Flask(__name__)

# Grab camera feed.
camera = cv2.VideoCapture(0)

# Gather cascade xml files from cv2.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def prep_frames():
    start_time = time.time()
    display_time = 1
    fps = 0

    while True:
        success, frame = camera.read()
        # The fllowing three lines can be used for screen capture if no webcam is available.
        # frame = pyautogui.screenshot()
        # frame = np.array(frame)
        # success = True

        if not success:
            break
        else:

            # Converts current frame to grayscale.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # For all faces detected draws an indigo rectangle.
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (130, 0, 75), 2)

            # For all eyes detected draws an orange rectangle.
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

            # Simple frame per second counter.
            fps += 1
            checkTime = time.time() - start_time
            if checkTime > display_time:
                fps = 0
                start_time = time.time()
            # Adds current fps to frame.
            cv2.putText(frame, str(int(fps / checkTime)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 185, 118), 2, cv2.LINE_AA)

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
