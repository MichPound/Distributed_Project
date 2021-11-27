# Code References:
    # https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask.
    # https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/.

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

def gen_frames():
    start_time = time.time()
    display_time = 2
    fps = 0

    # Setting up options for hand detection.
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils


    while True:
        success, frame = camera.read()
        # The fllowing three lines can be used for screen capture if no webcam is available.
        # frame = pyautogui.screenshot()
        # frame = np.array(frame)
        # success = True

        if not success:
            break
        else:
            # Processes the frame for hands.
            results = hands.process(frame)

            # Handles the drawin of hand feature landmarks.
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x *w), int(lm.y*h)
                        cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)

                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Renders the basic html page.
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(app.run(host="0.0.0.0"), debug=True)
