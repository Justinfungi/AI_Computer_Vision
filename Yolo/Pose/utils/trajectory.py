import cv2

def create_trajectory():
    trajectories=[]
    return trajectories


def modify_trajectory(trajectories,center):
    #if len(trajectories) >= 30:
        #del trajectories[0]

    trajectories.append(center)

    # intend to delete when scene change
    """if len(trajectories) != 0:
        print(center, trajectories[-2])
        if abs(center[0] - trajectories[-2][0]) >=25:
            print("clear")
            trajectories = []"""

    return 0


def plot_trajectory(trajectories, img):

    for j in trajectories[-30:-1]:
        if j == trajectories[-1]:
            cv2.circle(img, j, 4, (0,0,247), 4) # bgr
        else:
            index = trajectories.index(j)
            cv2.circle(img, j, 4, (10*index,25,19), 2)
    return 0
