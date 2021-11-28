# Distributed Systems

### Core Crumble Streaming
### Name: Michael Pound, 20085540

----------------------------------------------------------------------

### Commands:
	gunicorn --bind 0.0.0.0:5000 face_detection:app --timeout 300 --workers=2
	gunicorn --bind 0.0.0.0:5000 object_detection:app --timeout 300 --workers=2
	gunicorn --bind 0.0.0.0:5000 hand_detection:app --timeout 300 --workers=2

### Imports Needed:
    • sudo apt install python3-opencv
    • sudo pip install mediapipe
    • sudo pip install Flask
    • sudo apt-get install apache2 apache2-utils ssl-cert libapache2-mod-wsgi -y
    • sudo pip install gunicorn

### Technologies Used:
    • DroidCam 
    • OpenCV
    • YOLOv3
    • Media Pipe
    • Flask 
    • Gunicorn/WSGI

### NOTE!!
For this to work yolov3.weights will also need to be downloaded, this was to large to upload to GitHub.

This command should work for the needed files:
	wget https://pjreddie.com/media/files/yolov3.weights
