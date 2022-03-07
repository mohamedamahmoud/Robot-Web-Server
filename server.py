import cv2
import time
import threading
from flask import Response, Flask, request, render_template
import jyserver.Flask as jsf
import numpy as np
from flask import *  
# import ros_message as msg

# Image frame sent to the Flask object
global video_frame
video_frame = None
global video_frame2
video_frame2 = None
global hueLower, hueUpper, hue2Lower, hue2Upper, satHigh, satLow, valHigh, valLow
# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock
thread_lock = threading.Lock()
# Create the Flask object for the application


# load trained model
cvNet = cv2.dnn.readNetFromTensorflow('/home/mohamed/catkin_ws/src/robo_msg/scripts/frozen_inference_graph.pb', '/home/mohamed/catkin_ws/src/robo_msg/scripts/my.pbtxt')

classNames = ["background", "Ball", "Stop_sign"]
global xy_values
xy_values = []
global mode
mode = None

app = Flask(__name__)

@jsf.use(app)
class App:
    def send_message(self):
        print(f"**************The image pixels are")
    def moveup(self):
        msg.talker(int(msg.MoveType.FORWARD),1)

    def movedown(self):
        msg.talker(int(msg.MoveType.BACKWARD),1)

    def moveright(self):
        msg.talker(int(msg.MoveType.RIGHT),1)

    def moveleft(self):
        msg.talker(int(msg.MoveType.LEFT),1)

    def clearmove(self):
        msg.talker(int(msg.MoveType.STOP),1)

def nothing(x):
    pass

def ssd_detection(frame):
    rows = frame.shape[0]
    cols = frame.shape[1]
    totalArea = rows * cols
    # print(totalArea)
    # 640*480 = 307200

    cvNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    box_info = []
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            objectClass = int(detection[1])
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            area = (right - left) * (bottom - top)
            # print(area)
            ratio = area / totalArea

            box = [left, top, right, bottom, classNames[objectClass], score]
            box_info.append(box)

            #print(ratio)
            if 0.002 < ratio < 0.65:
                centerX = (left + right) / 2
                if centerX < 0.25 * cols:
                    print('turn left')
                    # msg.talker(int(msg.MoveType.LEFT),6)
                elif centerX > 0.75 * cols:
                    print('turn right')
                    # msg.talker(int(msg.MoveType.RIGHT),6)
                else:
                    if 0.04 < ratio <= 0.25:
                        # bounding box is covering less than 25% of total area
                        print('move forward at full speed')
                        # msg.talker(int(msg.MoveType.FORWARD),10)
                    elif 0.25 < ratio <= 0.40:
                        # bounding box is covering over 25% of total area
                        print('move forward slowly')
                        # msg.talker(int(msg.MoveType.FORWARD),4)
                    elif ratio > 0.60:
                        # bounding box is covering more than 60% of total area
                        print('move backward')
                        # msg.talker(int(msg.MoveType.BACKWARD),4)
                    else:  # object either too far away or too close to the camera
                        print('stop')
                        # msg.talker(int(msg.MoveType.STOP),0)
            else:
                # msg.talker(int(msg.MoveType.STOP),0)
                print('stop')
    return box_info


def hsv(frame):
    global hueLower, hueUpper, hue2Lower, hue2Upper, satHigh, satLow, valHigh, valLow
    global video_frame2
    counter = 0
    if counter < 11:
        counter = counter + 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array([hueLower, satLow, valLow])
        u_b = np.array([hueUpper, satHigh, valHigh])
        l_b2 = np.array([hue2Lower, satLow, valLow])
        u_b2 = np.array([hue2Upper, satHigh, valHigh])
        FGmask = cv2.inRange(hsv, np.float32(l_b), np.float32(u_b))
        FGmask2 = cv2.inRange(hsv, np.float32(l_b2), np.float32(u_b2))
        FGmaskComp = cv2.add(FGmask, FGmask2)
        contours, _ = cv2.findContours(FGmaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        video_frame2 = FGmask2.copy() # capture frames of the filtered live stream


def captureFrames2():
    global video_frame2, video_frame, thread_lock
    global hueLower, hueUpper, hue2Lower, hue2Upper, satHigh, satLow, valHigh, valLow
    # Initial values of the filters
    hueLower = 50
    hueUpper = 100
    hue2Lower = 50
    hue2Upper = 100
    satLow = 100
    satHigh = 255
    valLow = 100
    valHigh = 255
    box_info = []
    frame_count = 0
    # Video capturing from OpenCV
    cam = cv2.VideoCapture(0)
    width=cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camarea = width*height
    global mode
    while True and cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        if(mode == None):
            with thread_lock:
            # capturing the frames from the camera with no filters
                video_frame = frame.copy()
            frame_count += 1
        elif(mode == "HSV"):
            with thread_lock:
            # capturing the frames from the camera with no filters
                video_frame = frame.copy()
            frame_count += 1
            hsv(frame)
        elif(mode == "SSD"):
            with thread_lock:
            # capturing the frames from the camera with no filters
                video_frame = frame.copy()
            frame_count += 1
            # Create a copy of the frame and store it in the global variable,
            # with thread safe access
            if (frame_count >= 7):
                box_info = ssd_detection(frame)
                frame_count = 0
            if xy_values:
                # Mouse is clicked on the screen, and x y coordinate is sent
                # Create a rectangle with given coordinate at the center
                # Width of the rectangle is set to 90 pixels
                # Height of the rectangle is set to 100 pixels
                mouseLeft = xy_values[0] - 15
                mouseRight = xy_values[0] + 15
                mouseTop = xy_values[1] - 15
                mouseBottom = xy_values[1] + 15
                cv2.rectangle(frame, (int(mouseLeft), int(mouseTop)), (int(mouseRight), int(mouseBottom)),
                            (44, 178, 44), thickness=2)

            for box in box_info:
                if xy_values:
                    if (int(box[0]) <= mouseLeft) and (int(box[1]) <= mouseTop) and (int(box[2]) >= mouseRight) and (int(box[3])
                                                                                                            >= mouseBottom):

                        xy_values[0] = (int(box[0]) + int(box[2]))/2  # centerX = (left+right)/2
                        xy_values[1] = (int(box[1]) + int(box[3]))/2  # centerY = (top+bottom)/2
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (90, 150, 210), thickness=2)
                        cv2.putText(frame, str(box[4]) + ":" + "{:.2f}".format(box[5]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (53, 57, 174), 2)
                else:
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (90, 150, 210), thickness=2)
                        cv2.putText(frame, str(box[4]) + ":" + "{:.2f}".format(box[5]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (53, 57, 174), 2)

        else:
            with thread_lock:
            # capturing the frames from the camera with no filters
                video_frame = frame.copy()
            frame_count += 1


        
        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break    
    
    cam.release()



def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_image) + b'\r\n')

def encodeFrame2():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame2
            if video_frame2 is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame2)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_image) + b'\r\n')

    
        
@app.route("/video_feed")
def video_feed():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
    return Response(encodeFrame2(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/Validate', methods=['GET', 'POST'])
def thisRoute():
    information = request.data
    information = information.decode("utf-8")
    xy_values = information.split(",")#The first value is the x coordinates and the second is the y coordinate
    print(f"---------------------------X = {xy_values[0]}, Y = {xy_values[1]}")
    return information

@app.route('/filters', methods=['GET', 'POST'])
def filters():
    global hueLower, hueUpper, hue2Lower, hue2Upper, satHigh, satLow, valHigh, valLow
    information = request.data
    information = information.decode("utf-8")
    ftr = information.split(",")
    if(ftr[0] == "hueLower"):
        hueLower = ftr[1] 
        print(f"---------------------------hueLower = {hueLower}")
    elif(ftr[0] == "hueUpper"):
        hueUpper = ftr[1] 
        print(f"---------------------------hueUpper = {hueUpper}")
    elif(ftr[0] == "hue2Lower"):
        hue2Lower = ftr[1]
        print(f"---------------------------hue2Lower = {hue2Lower}")
    elif(ftr[0] == "hue2Upper"):
        hue2Upper = ftr[1]
        print(f"---------------------------hue2Upper = {hue2Upper}")
    elif(ftr[0] == "satLow"):
        satLow = ftr[1]
        print(f"---------------------------satLower = {satLow}")
    elif(ftr[0] == "satHigh"):
        satHigh = ftr[1] 
        print(f"---------------------------satHigh = {satHigh}")
    elif(ftr[0] == "valLow"):
        valLow = ftr[1]
        print(f"---------------------------valLow = {valLow}")
    elif(ftr[0] == "valHigh"):
        valHigh = ftr[1] 
        print(f"---------------------------valHigh = {valHigh}")

    return ""

@app.route('/Modes', methods=['GET', 'POST'])
def Modes():
    global mode
    information = request.data
    information = information.decode("utf-8")
    if(information == "controller"):
        mode = information
        print(f"---------------------------mode = {mode}")
    elif(information == "HSV"):
        mode = information 
        return redirect(url_for("xyz"))  
        print(f"---------------------------mode = {mode}")
    elif(information == "SSD"):
        mode = information
        print(f"---------------------------mode = {mode}")
    return ""

@app.route("/")
def index():
    return App.render(render_template("index.html")) 


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # Create a thread and attach the method that captures the image frames, to it 
    process_thread = threading.Thread(target=captureFrames2)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()
    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network
    app.run("0.0.0.0", port="5000")

