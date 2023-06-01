import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


Names =["Normal","Face", "Pose","Indentify","Follow"]

#Model Face parameters:
in_width = 300
in_height = 300
mean= [104,117,123]
conf_threshold = 0.7

#Model Pose parameters:
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)


#Build-in model import#
net_face = cv.dnn.readNetFromCaffe("../Model/deploy.prototxt",
                               "../Model/res10_300x300_ssd_iter_140000_fp16.caffemodel")

net_pose = cv.dnn.readNetFromCaffe("../Model/pose_deploy_linevec_faster_4_stages.prototxt",
                                "../Model/pose_iter_160000.caffemodel")


#sub funcs



'''
                            ## FaceReg
Activation: Press "F"/"f"
Input: frame of a video, window's named
Output: Face Recognition with landmarks on face when press a continously
'''
def FaceReg(frame,win_name):
    frame_height= frame.shape[0]
    frame_width= frame.shape[1]

    blob = cv.dnn.blobFromImage(
        frame, 1.0, (in_width, in_height),
        mean, swapRB= False,crop=False
        )
    # Run net_face
    net_face.setInput(blob)
    detections = net_face.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 0.5, 1)

            cv.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0))

            key= cv.waitKey(1)
            if key == ord("A") or key == ord("a"):
                with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    results = holistic.process(frame)
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                             )

    cv.imshow(win_name, frame)


'''
                        ## Pose Estimation
Activation: Press "P"/"p"
Input: frame of a video, window's named
Output: PoseEstimation with landmarks on hands and body
'''
def PoseEstimation(frame,win_name):
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )


         # 3. Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )



    cv.imshow(win_name, frame)






'''
                        # Main()
'''
def main():


    '''
                        #Setting Modes
    A list of modes listed and envalued
    '''
    Normal=0
    Face=1
    Pose=2
    Indentify=3
    Follow=4
    Modes =[Normal,Face, Pose,Indentify,Follow]


    '''
                        #Init Setting
    A list of modes listed and envalued
    '''
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]
    CurMode = Normal
    alive = True
    win_name = Names[0]
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    result = None
    capture = cv.VideoCapture(s)


    while alive:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        frame = cv.flip(frame,1)
        #t, _ = net_face.getPerfProfile()
        #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))




        # Mode Change and Key Manipulation
        #it is used to ocntrol the mode of the program by different keybinding

        if CurMode == Normal:
            win_name=Names[0]
            output = frame
            cv.imshow(win_name, output)

        elif CurMode == Face:
            win_name = Names[1]
            FaceReg(frame,win_name)

        elif CurMode == Pose:
            win_name = Names[2]
            PoseEstimation(frame, win_name)
        elif CurMode == Indentify:
            break
        elif CurMode == Follow:
            break

        key = cv.waitKey(1)
        if key == ord('Q') or key == ord('q') or key == 27:
            alive = False
        elif key == ord('F') or key == ord('f'):
            cv.destroyWindow(win_name)
            CurMode = Face
        elif key == ord('N') or key == ord('n'):
            cv.destroyWindow(win_name)
            CurMode = Normal
        elif key == ord('P') or key == ord('p'):
            cv.destroyWindow(win_name)
            CurMode = Pose


    '''

        elif key == ord('P') or key == ord('p'):
            CurMode = PREVIEW'''



    source.release()
    cv.destroyWindow(win_name)




main()
