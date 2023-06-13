import cv2
from math import sqrt
import pandas as pd
import numpy as np

def MA(array):
    sum_ = 0
    if len(array) >= 10:
        for i in range(-1,-10,-1):
            sum_ += array[i]
        avg_ = int(sum_/10)   # reduce fluctuation of speed
    else:
        avg_ = array[-1]

    if array[-1] == 0:
        avg_ = 0  # to detect change direction

    return avg_

def create_array():
    velocity_array =[]
    acceleration_array = []
    avg_velocity_array = []
    avg_acceleration_array = []
    return velocity_array, acceleration_array, avg_velocity_array, avg_acceleration_array

def store_array(array,source):
    pd.DataFrame(np.array(array)).to_excel(f"data/excel/array_{source[13:]}_output.xlsx")
    return 0

def get_velocity(trajectory, img, xyxy, velocity_array, avg_velocity_array):

    cv2.rectangle(img, (50,150),(70,350),(0,255,0),3)
    cv2.putText(img, "Velocity", (25,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 2)

    h, w, _ = img.shape
    player_height = abs(int((xyxy[1]-xyxy[3])/2)) # normally is 86

    if len(trajectory) >=3:
        """if trajectory[-3][0] <= trajectory[-2][0] and trajectory[-2][0] >= trajectory[-1][0]:
            velocity = 0
        elif trajectory[-3][0] >= trajectory[-2][0] and trajectory[-2][0] <= trajectory[-1][0]:
            velocity = 0
        else:
            """
        if trajectory[-2][0] == trajectory[-1][0]:
            velocity = 0
        else:
            velocity = sqrt((trajectory[-2][0]-trajectory[-1][0])**2 + (trajectory[-2][1]-trajectory[-1][1])**2)

        velocity = int(velocity/(w/5) * 2000 ) # standardization for court  ---> 5 division of the court
        if velocity >= 500:
            velocity = 0 # remove big scene velocity

        #print(player_height , h/4)
        if player_height >= h/5:
            velocity_array.append(0)
            avg_velocity_array.append(0)
            text = "* It is not on the game"
            cv2.rectangle(img, (0,0),(400,30),(0,0,0),cv2.FILLED)
            cv2.putText(img, text, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,122,255), 2)

        else:
            velocity_array.append(velocity)
            avg_velocity = MA(velocity_array)
            avg_velocity_array.append(avg_velocity)
            new = 350-avg_velocity
            #print("vel",avg_velocity)
            if new <=150:
                new = 150
            # 250 pixel space
            if avg_velocity <= 84:
                cv2.rectangle(img, (50,new),(70,350),(255,255,255),cv2.FILLED)
            elif avg_velocity <= 130:
                cv2.rectangle(img, (50,new),(70,350),(0,160,247),cv2.FILLED)
            else:
                cv2.rectangle(img, (50,new),(70,350),(33,37,227),cv2.FILLED)

    return velocity_array, avg_velocity_array

def get_acceleration(velocity_array, acceleration_array, img, xyxy, avg_acceleration_array):

    cv2.rectangle(img, (50,475),(70,675),(0,255,0),3)
    cv2.putText(img, "Acceleration", (25,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 2)


    if len(velocity_array) >=2:
        acceleration = abs(velocity_array[-1] -  velocity_array[-2])
        #print("acceleration",acceleration)

        h, w, _ = img.shape
        player_height = abs(int((xyxy[1]-xyxy[3])/2))

        if player_height >= h/4:
            acceleration_array.append(0)
            avg_acceleration_array.append(0)
            text = "* It is not on the game"
            cv2.rectangle(img, (0,0),(400,30),(0,0,0),cv2.FILLED)
            cv2.putText(img, text, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,122,255), 2)

        else:
            acceleration_array.append(acceleration)
            avg_acceleration = MA(acceleration_array)
            avg_acceleration_array.append(avg_acceleration)

            cv2.rectangle(img, (50,675-avg_acceleration),(70,675),(108,35,56),cv2.FILLED)

    return acceleration_array, avg_acceleration_array

# Using COD Deficit (CODD) to Assess Agility
# The COD deficit for the 505 for each leg was calculated by the formula: mean 505 time − mean 10-m time ( 27).

def all_zero(lst):
    lst = np.array(lst)
    if np.any(lst):
        return True # not all zero
    else:
        return False # all zero

def find_t2(lst, condition, Left_side):
    t2=0
    lst.reverse()
    if Left_side == True:
        for i in lst:
            if i[0] >= condition:
                t2 = lst.index(i)
                break
    else:
        for i in lst:
            if i[0] <= condition:
                t2 = lst.index(i)
                break
    return t2

def find_t34(lst, condition, Left_side):
    t34 = 0
    lst.reverse()
    if Left_side == True:
        for i in lst:
            if i[0] <= condition:
                t34 = lst.index(i)
                break
    else:
        for i in lst:
            if i[0] >= condition:
                t34 = lst.index(i)
                break
    return t34 # It seems buggy

def Agility_calculation(t1,t2,t3,t4):
    if t2==0:
        t2 = t4

    sprint = (t4+t2)/2
    turn = (t3+t1)

    try:
        Agility_result = (sprint - turn) / sprint
    except:
        Agility_result = 1

    if Agility_result > 1 or Agility_result < -1:
        Agility_result=0
    return Agility_result

def get_agility(trajectory, avg_velocity_array, img, Agility_testing):
    h, w, _ = img.shape

    """print(avg_velocity_array[-30:-1])
    print(Agility_testing.testing_signal,"Signal")
    print(Agility_testing.testing_round,"Round")
    print(avg_velocity_array[-30:-1])"""
    # Bar
    cv2.putText(img, f"Agility", (w-70,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 2)
    cv2.putText(img, f"{round(Agility_testing.result,2)}", (w-70,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 1)
    cv2.rectangle(img, (w-60,150),(w-40,350),(0,255,0),3)

    cv2.rectangle(img, (w-60,150),(w-40,250),(60,22,128),cv2.FILLED) # Above
    cv2.rectangle(img, (w-60,250),(w-40,350),(101,21,21),cv2.FILLED) # Below
    agility_value = 350-100- int(Agility_testing.result * 100)
    cv2.rectangle(img, (w-70,agility_value-1),(w-30,agility_value+1),(255,255,255),cv2.FILLED)

    # Update x1,x2,x3, t2
    if len(avg_velocity_array) >= 10:
        if avg_velocity_array[-1] == 0 and all_zero(avg_velocity_array[-5:])  :
         # changeofdirection ; can consider only detection done then enter this part
            Agility_testing.testing_signal = True
            Agility_testing.x1 = trajectory[-1][0]
            Agility_testing.x2 = trajectory[-10][0]

            turn_distance = abs(Agility_testing.x1 - Agility_testing.x2) # S1 ==> t1 = 10
            sprint_distance = turn_distance * 2

            # Left side - Right_side test
            Agility_testing.Left_side = Agility_testing.x2 > Agility_testing.x1

            if Agility_testing.Left_side == True:
                Agility_testing.x3 = Agility_testing.x2 + sprint_distance
            else:
                Agility_testing.x3 = Agility_testing.x2 - sprint_distance

            Agility_testing.t2 = find_t2(trajectory[-30:], Agility_testing.x3, Agility_testing.Left_side)

    if Agility_testing.x3 >= w or Agility_testing.x1 <= 0:
        Agility_testing.testing_signal = False

    if Agility_testing.testing_signal == True:
        text2 = f"* 505 Agility Test - Round {Agility_testing.testing_round} Start"
        cv2.rectangle(img, (w-300,10),(w,35),(0,0,0),cv2.FILLED)
        cv2.putText(img, text2, (w-300,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,122,255), 1)

        #print(Agility_testing.x1, Agility_testing.x2, Agility_testing.x3)
        if Agility_testing.Left_side == True and trajectory[-1][0] >= Agility_testing.x3:
            Agility_testing.t4 = find_t34(trajectory[-30:], Agility_testing.x2, Agility_testing.Left_side)
            Agility_testing.t3 = find_t34(trajectory[-30:], Agility_testing.x1, Agility_testing.Left_side) - Agility_testing.t4

            Agility_testing.result = Agility_calculation(Agility_testing.t1,Agility_testing.t2,Agility_testing.t3,Agility_testing.t4)

            """print(trajectory[-30:])
            print(Agility_testing.t1, Agility_testing.t2, Agility_testing.t3, Agility_testing.t4,"t34")
            print(sprint, turn)
            print("/"*100)
            print(Agility_testing.result, "Result")"""

            Agility_testing.testing_signal = False
            Agility_testing.testing_round += 1

        elif Agility_testing.Left_side == False and trajectory[-1][0] <= Agility_testing.x3:
            Agility_testing.t4 = find_t34(trajectory[-30:], Agility_testing.x2, Agility_testing.Left_side)
            Agility_testing.t3 = find_t34(trajectory[-30:], Agility_testing.x1, Agility_testing.Left_side) - Agility_testing.t4

            Agility_testing.result = Agility_calculation(Agility_testing.t1,Agility_testing.t2,Agility_testing.t3,Agility_testing.t4)

            """print(trajectory[-30:])
            print(Agility_testing.t1, Agility_testing.t2, Agility_testing.t3, Agility_testing.t4,"t34")
            print(sprint, turn)"""

            Agility_testing.testing_signal = False
            Agility_testing.testing_round += 1

    elif  Agility_testing.testing_signal == False:
        text2 = f"* 505 Agility Test - Round {Agility_testing.testing_round-1} Finished"
        cv2.putText(img, text2, (w-325,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,122,255), 1)




    return Agility_testing

def GameStart(avg_velocity_array):
    if all_zero(avg_velocity_array[-10:-2]) == False and avg_velocity_array[-1] != 0:
        return True
    else:
        return False

def GameEnd(avg_velocity_array):
    if all_zero(avg_velocity_array[-9:-1]) == False and avg_velocity_array[-10] != 0:
        return True
    else:
        return False

def GameMeanCaculation(avg_velocity_array):
    return sum(avg_velocity_array)/ len(avg_velocity_array)

def PD_MeanVelocity(GameMean, lastmean):
    return (GameMean - lastmean)/lastmean

def get_endurance(avg_velocity_array,img,Endurance_testing):
    h, w, _ = img.shape
    endurance_value = 675-int(Endurance_testing.result * 100)

    cv2.rectangle(img, (w-60,475),(w-40,675),(0,255,0),3) # Outer box
    cv2.rectangle(img, (w-60,475),(w-40,575),(60,22,128),cv2.FILLED) # Above
    cv2.rectangle(img, (w-60,575),(w-40,675),(101,21,21),cv2.FILLED) # Below
    cv2.rectangle(img, (w-70,endurance_value-1),(w-30,endurance_value+1),(255,155,255),cv2.FILLED) # White indicator


    cv2.putText(img, "Endurance", (w-100,440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 2)
    cv2.putText(img, f"{round(Endurance_testing.result,2)}", (w-70,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,37,227), 2)

    if len(avg_velocity_array) >= 10:
        Endurance_testing.GameStart = GameStart(avg_velocity_array)
        Endurance_testing.GameEnd = GameEnd(avg_velocity_array)
        print(Endurance_testing.GameStart, Endurance_testing.GameEnd)

    print(avg_velocity_array[-20:])
    if Endurance_testing.GameStart:
        Endurance_testing.GameStartIndex = len(avg_velocity_array)
        print(Endurance_testing.GameStartIndex) # 61 is the first
        print( "/"*200)

    if Endurance_testing.GameEnd:
        GameMean = GameMeanCaculation(avg_velocity_array[Endurance_testing.GameStartIndex:])
        Endurance_testing.GameCount +=1
        print(Endurance_testing.GameCount)
        print( "/"*200)

        if Endurance_testing.GameCount != 1 and GameMean>= 10 and Endurance_testing.lastmean !=0:
            print(GameMean, Endurance_testing.lastmean)
            Endurance_testing.result = Endurance_testing.result * (1+PD_MeanVelocity(GameMean, Endurance_testing.lastmean))
            #if Endurance_testing.result >1 or Endurance_testing.result >1

        if GameMean !=0:
            Endurance_testing.lastmean = GameMean
        print(GameMean, "#"*100)

    return Endurance_testing
