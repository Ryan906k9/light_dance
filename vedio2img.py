import cv2

cap = cv2.VideoCapture("cxk.mp4")
i=1
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    else:
        cv2.imwrite("cxk_images/"+str(i)+".jpg",frame)
        print(i)
        i+=1