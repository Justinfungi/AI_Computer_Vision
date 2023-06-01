# import the library
import cv2 as cv
import matplotlib.pyplot as plt

source = './Videos/race_car.mp4'  # source = 0 for webcam

cap = cv.VideoCapture(source)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

ret, frame = cap.read()

while True:
    istrue, frame = cap.read()
    cv.imshow("Video",frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break







'''
Write Video using OpenCV
For writing the video, you need to create a videowriter object with the right parameters.

Function Syntax
VideoWriter object=   cv.VideoWriter( filename, fourcc, fps, frameSize  )
where, Parameters

filename: Name of the output video file.
fourcc: 4-character code of codec used to compress the frames.
For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec,
VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.
List of codes can be obtained at Video Codecs by FOURCC page.
FFMPEG backend with MP4 container natively uses other values as fourcc code:
    see ObjectType, so you may receive a warning message from OpenCV about
    fourcc code conversion.
fps: Framerate of the created video stream.
frameSize: Size of the video frames.
'''

# Default resolutions of the frame are obtained.
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.
out_avi = cv2.VideoWriter('race_car_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

out_mp4 = cv2.VideoWriter('race_car_out.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:

    # Write the frame to the output files
    out_avi.write(frame)
    out_mp4.write(frame)

  # Break the loop
  else:
    break

cap.release()
out_avi.release()
out_mp4.release()

cv.destroyAllWindows
