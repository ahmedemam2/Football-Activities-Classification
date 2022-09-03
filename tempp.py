import cv2
import os
import shutil
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from dollarpy import Template,Recognizer,Point
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

dictAccuracy = {
    "juggle": 0,
    "dribble": 0,
    "shoot": 0
}
def windowing(seconds,path,arr):
    trainvideo = arr + "1"
    temp = 0
    for i in range(seconds):
        index = temp
        x = trainvideo[:-1] + str(index) + '.mp4'
        ffmpeg_extract_subclip(path, temp, temp+1, targetname=x)
        # moving to videos folder
        src =  "" + x
        dst = r"Windows" + "/" + arr + "/" + x
        shutil.move(src, dst)
        temp += 1

def getactivityduration(path,arr):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    seconds = int(duration%60)
    print(seconds)
    windowing(seconds,path,arr)
    cap.release()

activities = []
for arr in os.listdir("videos"):
    path = "videos" + '/' + arr
    arr = arr.removesuffix('.mp4')
    activities.append(arr)
    getactivityduration(path,arr)
def getTrainlandmarks(path,target):
    cap = cv2.VideoCapture(path)
    xl = []
    yl = []
    templ = []
    templx = []
    temply = []
    labellist = []
    print(path)
    while True:
        success, img = cap.read()
        success, frames = cap.read()
        try:
            imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        except:
            break
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    xl.append(lm.x)
                    yl.append(lm.y)
                    # templ.append([lm.x,lm.y])
                    templ.append(lm.x)
                    templ.append(lm.y)

        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    # templ.append((xl,yl))
    labellist.append(target)
    return templ,labellist

landmarksT , landmarkstemp = [] , []
labelT , labeltemp = [] , []
landmarksl = []
labell = []
for i in range(len(activities)):
    for file in os.listdir("Windows" + "/" + activities[i]):
        path = "Windows" + '/' + activities[i] + "/" + file
        arr = arr.removesuffix('.mp4')
        landmarkstemp , labeltemp = getTrainlandmarks(path,activities[i])
        landmarksl.append(landmarkstemp)
        labell.append(labeltemp)
print(len(labell))
print(len(landmarksl))
df = pd.DataFrame({
    "label": landmarksl,
    "prediction" : labell
})
df.to_csv(r'haaa.csv',index=None)
X_test , temp = getTrainlandmarks("test/finaltest.mp4",None)

X_train = []

X_train = landmarksl
y_train = labell
np_X_Train = np.array(X_train)
np_Y_Train = np.array(y_train)
np_X_Test = np.array(X_test)
listtargets = []
knn = KNeighborsClassifier(n_neighbors=3)



knn.fit([np_X_Train]*26  ,
        np_Y_Train)



prediction = knn.predict([np_X_Test]*26)
confidence = knn.predict_proba([np_X_Test]*26)
print(confidence)
testlist = []
templist = []
for i in range(len(confidence)):
    testlist.append(confidence[i])
    maxx = max(confidence[i][0],confidence[i][1],confidence[i][2])
    if maxx > 0.3:
        print(maxx)
        templist.append(maxx)
flagct = 0
print(len(confidence))
print(len(templist))
df = pd.DataFrame({
    "prediction" : testlist,
    "label": prediction,
    "max": templist
})
df.to_csv(r'TEMP.csv', index=False)
for i in range(len(prediction)):
    if prediction[i] == "shoot":
        dictAccuracy["shoot"] += 1
    if prediction[i] == "juggle":
        dictAccuracy["juggle"] += 1
    if prediction[i] == "dribble":
        dictAccuracy["dribble"] += 1
    if flagct == 32:
        flagct = 0
        if dictAccuracy["shoot"] > dictAccuracy["juggle"] and dictAccuracy["shoot"] > dictAccuracy["dribble"]:
            fin_max = "shoot"
        if dictAccuracy["dribble"] > dictAccuracy["shoot"] and dictAccuracy["dribble"] > dictAccuracy["juggle"]:
            fin_max = "dribble"
        if dictAccuracy["juggle"] > dictAccuracy["shoot"] and dictAccuracy["juggle"] > dictAccuracy["dribble"]:
            fin_max = "juggle"
        dictAccuracy["juggle"] = 0
        dictAccuracy["dribble"] = 0
        dictAccuracy["shoot"] = 0
        # fin_max = max(dictAccuracy, key=dictAccuracy.get)
        # print("Maximum value:",fin_max)
        listtargets.append(fin_max)
    flagct += 1
cap = cv2.VideoCapture("finaltest.mp4")
ct = 0
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        ct = ct+1
        text = listtargets[ct]
        print(text)
        img = cv2.putText(img, text, (00, 185), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
