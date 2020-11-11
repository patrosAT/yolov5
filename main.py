import time
import cv2
import torch

from detector import YOLOv5Detector


detector = YOLOv5Detector()
print(detector.names[41])

#image = cv2.imread('../examples/003.jpg')
cap = cv2.VideoCapture('../examples/M6_Motorway_Traffic.mp4')




while(True):
    _, frame = cap.read()
    det = detector.detect(frame)

    for i, res in enumerate(det):
        cv2.rectangle(frame, (int(res[0]), int(res[1])),
                             (int(res[2]), int(res[3])),
                             (255, 0, 0),
                             2,
                             lineType=cv2.LINE_8)   

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()








'''
cap = cv2.VideoCapture('examples/street.mp4')

while(True):
    # Capture frames
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Our operations on the frame come here
    with torch.no_grad():
        res = detect(image)
        print(res)
        print('-----------------')
    

    # Display the resulting frame
    frame_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''

