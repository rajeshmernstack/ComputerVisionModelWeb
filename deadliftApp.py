
from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
from numpy import savetxt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class poseDetector():
    def __init__(self, mode=False, modelComp = 1, smooth=True, enable_seg=False,
                    smooth_seg=False, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComp = modelComp
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smooth, self.enable_seg, self.smooth_seg,
                                        self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                IsPose = True
        else:
                IsPose = False
        return img, IsPose

    def findPosition(self, img, draw=True):
        self.lmList = []
        self.world = self.results.pose_world_landmarks.landmark
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                #if draw:
                #    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return self.lmList, self.world

    def findAngle(self, img, p1, p2, p3, draw=False):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        joint1 = np.array([self.world[p1].x, self.world[p1].y, self.world[p1].z])
        joint2 = np.array([self.world[p2].x, self.world[p2].y, self.world[p2].z])
        joint3 = np.array([self.world[p3].x, self.world[p3].y, self.world[p3].z])

        # Calculate the Angle
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(joint1-joint2, joint3-joint2)),
                                            np.dot(joint1-joint2, joint3- joint2)))

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

        return angle

    def angle2d(self, img, p1, p2, p3, draw=False):
        # Get the landmarks
        xy1 = np.array(self.lmList[p1][1:])
        xy2 = np.array(self.lmList[p2][1:])
        xy3 = np.array(self.lmList[p3][1:])


        # Calculate the Angle
        angle = np.rad2deg(np.arctan2(np.linalg.det([xy1-xy2, xy3-xy2]), np.dot(xy1-xy2, xy3- xy2)))

        # print(angle)

        # Draw
        if draw:
            cv2.putText(img, str(int(angle)), xy2,
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)


        return np.abs(angle)



# cap = cv2.VideoCapture(0)

calib = [0,0,0,0]
calibIdx = 0
disKnee = []
calibNose = [320, 100]
calibRHeel = [290, 420]
calibLHeel = [340, 420]
calbRes = 35
status = 0
countFrame = 0
repDown = 0
repUp = 1
rep = 0

arrowLR = cv2.imread("arrowLR_no_bg.png") # =>
arrowRL = cv2.imread("arrowRL_no_bg.png") # <=
arrowUp = cv2.imread("arrowUp.png")
arrowLR = cv2.resize(arrowLR,(30,30))
arrowRL = cv2.resize(arrowRL,(30,30))
arrowUp = cv2.resize(arrowUp,(30,30))


detector = poseDetector()


# while cap.isOpened():
# while True:
# success, img = cap.read()
img = cv2.imread("old.jpg")
img = cv2.flip(img, 1)
img, IsPose = detector.findPose(img, draw=True)
# frame_width = int(cap.get(3)) # 640
# frame_height = int(cap.get(4)) # 480


if IsPose:
    lmList, world = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        ## Joint angle of interest
        # Left Shoulder
        RightShoulderAng = detector.findAngle(img, 13, 11, 23)
        # Left Elbow
        RightElbowAng = detector.findAngle(img, 15, 13, 11)
        # Left Hip
        RightHipAng = detector.angle2d(img, 11, 23, 25, False)
        # Left Knee
        RightKneeAng = detector.angle2d(img, 23, 25, 27, False)
        # Left back 
        RightbackAng = detector.angle2d(img, 7, 11, 23, False)

        # Right Shoulder
        LeftShoulderAng = detector.findAngle(img, 14, 12, 24)
        # Right Elbow
        LeftElbowAng = detector.findAngle(img, 16, 14, 12)
        # Right Hip
        LeftHipAng = detector.angle2d(img, 12, 24, 26, False)
        # Right Knee
        LeftKneeAng = detector.angle2d(img, 24, 26, 28, False)
        # Right back 
        LeftbackAng = detector.angle2d(img, 8, 12, 24, False)
        

    if status == 0:
        xRShoulder,yRShoulder = lmList[24][1:]
        xLShoulder,yLShoulder = lmList[23][1:]
        xRHip,yRHip = lmList[26][1:]
        xLHip,yLHip = lmList[25][1:]
        xRKnee,yRKnee = lmList[26][1:]
        xLKnee,yLKnee = lmList[25][1:]
        xRToe,yRToe = lmList[32][1:]
        xLToe,yLToe = lmList[31][1:]


        cv2.line(img,(xLShoulder,450),(xLShoulder+50,450),(255,0,0),3)

        if np.abs(xRShoulder-xRHip) <= 25:
            calib[0] = 1
            cv2.line(img,(xRShoulder,450),(xRShoulder-50,450),(255,0,0),3)
        else:
            calib[0] = 0
            cv2.line(img,(xRShoulder,450),(xRShoulder-50,450),(0,255,0),3)

        if np.abs(xLShoulder-xLHip) <= 25:
            calib[1] = 1
            cv2.line(img,(xLShoulder,450),(xLShoulder-50,450),(255,0,0),3)
        else:
            calib[1] = 0
            cv2.line(img,(xLShoulder,450),(xLShoulder-50,450),(0,255,0),3)


        if np.abs(xRShoulder-xRKnee) <= 25:
            calib[2] = 1
            cv2.line(img,(xRHip,yRHip),(xRKnee,yRKnee),(255,0,0),3)
        else:
            calib[2] = 0
            cv2.line(img,(xRHip,yRHip),(xRKnee,yRKnee),(0,255,0),3)

        if np.abs(xLShoulder-xLKnee) <= 25:
            calib[3] = 1
            cv2.line(img,(xLHip,yLHip),(xLKnee,yLKnee),(255,0,0),3)
        else:
            calib[3] = 0
            cv2.line(img,(xLHip,yRHip),(xLKnee,yLKnee),(0,255,0),3)
        if np.sum(calib) == 4:
            calibIdx += 1
            startTime = time.time()
        else:
            calibIdx = 0
            calib[0] = 0
            calib[1] = 0
            disKnee = []

        if calibIdx >= 1:
            cv2.putText(img,"Stay there",(10,100),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),5)
            disKnee.append(np.abs(xRKnee - xLKnee))

        if calibIdx >= 100:
            status = 1

    if status == 1:
        xRKnee,yRKnee = lmList[26][1:]
        xLKnee,yLKnee = lmList[25][1:]
        xRShoulder,_ = lmList[12][1:]
        xLShoulder,_ = lmList[11][1:]
        xLEar,yLEar = lmList[7][1:]

        currentTime = time.time()

        countFrame += 1

        if RightKneeAng < 100 and xRKnee < xRHip:
            try:
                img[yRKnee-15:yRKnee+15,xRKnee-15:xRKnee+15,:] = arrowLR
            except:
                pass

        
        if LeftKneeAng < 100 and xLKnee > xLHip:
            try:
                img[yLKnee-15:yLKnee+15,xLKnee-15:xLKnee+15,:] = arrowRL
            except:
                pass

        if LeftbackAng < 160:
            try:
                img[yLEar-15:yLEar+15,xLEar-15:xLEar+15,:] = arrowUp
            except:
                pass

        if RightHipAng <= 75 and repDown == 0:
            repDown = 1
            repUp = 0

        if RightHipAng >= 170 and repUp == 0:
            repDown = 0
            repUp = 1
            rep += 1

        cv2.putText(img, str(rep), (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,255),5)
    # else:
    #     continue

    countFrame += 1
    CountAngle = []

    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imwrite("new.jpg", img)
    # cv2.imshow('Image', img)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


# cap.release()
# cv2.destroyAllWindows()
cv2.destroyAllWindows()


