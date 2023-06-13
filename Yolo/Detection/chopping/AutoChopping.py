import cv2
import pandas as pd
import argparse
import os

def check_folder():
    dir_path = f"{opt.source}"
    if not os.path.exists(dir_path):
        # if the folder doesn't exist, create a new one
        os.makedirs(dir_path)
        print("New folder created: {}".format(dir_path))
    else:
        print("Folder already exists: {}".format(dir_path))

def chopping():
    df = pd.read_excel(f'{opt.source}.xlsx')
    lst = df.values.tolist()[0]
    startindex = []
    endindex = []

    for i in range(1,len(lst)) :
            if lst[i] ==1 and lst[i-1] ==0 :
                startindex.append(i)
            elif lst[i] ==0 and lst[i-1] ==1 :
                endindex.append(i)

    pairs = list(zip(startindex, endindex))
    merged_pairs = [(pairs[0][0],pairs[0][1],1)]
    for i in range(1,len(pairs)):
        start_prev, end_prev, label = merged_pairs[-1]
        start, end = pairs[i]
        try:
            start_next, end_next = pairs[i+1]
        except:
            start_next, end_next = 999999,999999

        distance1 = start - merged_pairs[-1][1]
        distance2 = start_next - end
        distance_inner = merged_pairs[-1][1] - merged_pairs[-1][0]

        print("merge1", start, end)
        if distance1 <= 30: # define close
            merged_pairs.append((start, end, label))
            #print("enter1")

        elif distance_inner <= 25:
            if distance1 < distance2:
                del merged_pairs[-1]
                merged_pairs.append((start_prev, end, label))
                #print("enter2")
            else:
                merged_pairs.append((start, end, label))
                #print("enter2")
        else:
            merged_pairs.append((start, end, label+1))
            #print("enter3")

    print(merged_pairs)

    merged_pairs2 = [merged_pairs[0]]
    for i in range(1,len(merged_pairs)):
        if merged_pairs2[-1][-1] != merged_pairs[i][-1]:
            merged_pairs2.append(merged_pairs[i])
        else:
            start, end, label = merged_pairs2[-1]
            start_new, end_new, label_new = merged_pairs[i]
            del merged_pairs2[-1]
            merged_pairs2.append((start, end_new, label))

    print("merged_pairs2",merged_pairs2)

    check_folder()
    counter = 1
    for (i,j,k) in merged_pairs2:
        print(i,j,k)
        cap = cv2.VideoCapture(f'{opt.source}.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = i
        end_frame = j

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 创建VideoWriter对象，用于将剪辑后的视频写入到新的视频文件中
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{opt.source}/Rally {counter}.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        counter +=1
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

        # 释放VideoCapture和VideoWriter对象
        cap.release()
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Demo2.xlsx', help='source path(s)')
    parser.add_argument('--folder', type=str, default='somebody', help='foler name(s)')
    opt = parser.parse_args()
    print(opt)
    chopping()
