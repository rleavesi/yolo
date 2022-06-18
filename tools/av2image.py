import cv2
videoFile = '../2021-RMUC-SZGS-CB02.mp4'
c = 9000
timeF = 10  #视频帧计数间隔次数

outputFile = './images/'
vc = cv2.VideoCapture(videoFile)
if vc.isOpened():
    rval, frame = vc.read()
else:
    print('openerror!')
    rval = False


while rval:
    print(c)
    rval, frame = vc.read()
    if c % timeF == 0:
        cv2.imwrite(outputFile + str(int(c)) + '.jpg', frame)
    c += 1
    cv2.waitKey(1)
vc.release()

