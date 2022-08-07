import mediapipe as mp
import cv2
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

cap=cv2.VideoCapture(0)

lcounter=0
lstage=None

rcounter=0
rstage=None


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

with mp_pose.Pose(min_tracking_confidence=0.8,min_detection_confidence=0.8) as pose:
    while cap.isOpened():
        ret,frame=cap.read()

        #frame=cv2.flip(frame,1)

        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        results=pose.process(image)

        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks=results.pose_landmarks.landmark
            #PUSHUPS
            left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            langle=calculate_angle(left_elbow,left_shoulder,left_hip)
            rangle=calculate_angle(right_elbow,right_shoulder,right_hip)

            cv2.putText(image, str(langle),
                        tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(rangle),
                        tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            if langle<22 or rangle<22:
                lstage="Down"

            if (langle>55 and lstage=="Down") or (rangle>55 and rstage=="Down"):
                lstage="Up"
                lcounter+=1


        except:
            pass

        cv2.rectangle(image,(0,0),(225,75),(245,117,16),-1)
        cv2.putText(image,'REPS',(12,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,10,0),1,cv2.LINE_AA)
        cv2.putText(image,str(lcounter),(13,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,10,0),1,cv2.LINE_AA)
        cv2.putText(image,'STAGE',(120,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,10,0),1,cv2.LINE_AA)
        cv2.putText(image,str(lstage),(125,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,10,0),1,cv2.LINE_AA)

        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Push-ups",image)

        if cv2.waitKey(10)&0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()