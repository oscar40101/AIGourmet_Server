from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
import os

global food_list, food_dic # 裝偵測到的食物
global frame
global stopframe
food_list = []
food_dic = {}
frame = None
stopframe = 1

answerlist=['potato','eggplant'] # 自定義只能框哪些答案

def yolowebcam():
    ap = argparse.ArgumentParser()
    # # ap.add_argument("-y", "--yolo", default='yolo-coco',
    # # help="base path to YOLO directory")
    ap.add_argument("-o", "--output", default=False,
    help="path to output video")
    # # ap.add_argument("-c", "--confidence", type=float, default=0.5,
    # # help="minimum probability to filter weak detections")
    # # ap.add_argument("-t", "--threshold", type=float, default=0.3,
    # # help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.join('./yolo-coco', "obj.names")
    LABELS = open(labelsPath).read().strip().split("\n")

    # print(len(LABELS))

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # print(COLORS)

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.join('./yolo-coco', "yolov4-tiny-obj-20210204-2_last.weights")
    configPath = os.path.join('./yolo-coco', "yolov4-tiny-obj-20210204-2.cfg")

    # load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()

    # print(ln)

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # # print(ln) #['yolo_30', 'yolo_37']

    # check if the video writer is enabled
    if args["output"] is not False:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 2,(640, 360), True)

    (W, H) = (None, None)
    print("[INFO] starting video capture...")
    cap = VideoStream(src=0,apiPreference=cv2.CAP_V4L).start() # 將webcam打開
    #cap = VideoStream(usePiCamera=True).start() 樹梅派用
    time.sleep(2.0)


    while True:
        frame = cap.read()
        frame = cv2.resize(frame, (640, 360)) # 輸出frame大小，跟呈現在html上一樣

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward pass of the YOLO object detector,
        # giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        # start = time.time()
        layerOutputs = net.forward(ln)
        # end = time.time()

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []


        for output in layerOutputs:     # loop over each of the layer outputs
            for detection in output:        # loop over each of the detections
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
            0.3)

        
       

        

        # 用來判斷是否停止分析
        if frame != []:
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten(): 
                    food_detect = LABELS[classIDs[i]]  # 預測項目名稱

                    if food_detect in answerlist: # 只有answer可以顯示出來
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #在圖片上標註位置，畫長方形
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                            confidences[i])
                        cv2.putText(frame, text, (x, y - 5),  #在圖片上標註預測結果和信心指數
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        
                        print(LABELS[classIDs[i]]) # 預測項目名稱
                        print(confidences[i]) # 預測信心指數

                        if food_detect not in food_list:
                            food_list.append(food_detect) # 將偵測項目加入food_list
                        yield frame # 此frame就是yolo分析出來，將最終結果畫在上面的那一張圖片，透過yield不斷回傳新的照片，做到串流的效果


        if args["output"] is not False:
            writer.write(frame)

        cv2.imshow("Output", frame) # 
        if cv2.waitKey(1) & stopframe == 0: # 當user案stop時 因stopframe會等於0 所以停止分析
            break
        if cv2.waitKey(1) & 0xFF == ord('q'): # 當user案q時結束分析
            break
        
    print(food_list)
    
    print("[INFO] cleanup up...")
    if args["output"] is not False:
        writer.release()

    # release the video stream pointer
    cap.stop()

def yoloresult():
    global food_list
    return food_list

def yolostop(): # 案stop結束分析
    global stopframe
    global food_list
    stopframe = 0
    return food_list
    