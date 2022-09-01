import cv2
import os
import shutil
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
    for i in range(seconds-1):
        temp += 1
        index = temp
        x = trainvideo[:-1] + str(index) + '.mp4'
        ffmpeg_extract_subclip(path, temp, temp+1, targetname=x)
        # moving to videos folder
        src = r"C:/Users/Ahmed/PycharmProjects/MediaPipe" + "/" + x
        dst = r"C:/Users/Ahmed/PycharmProjects/MediaPipe/Windows" + "/" + arr + "/" + x
        shutil.move(src, dst)

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
    labellist = []
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
                    labellist.append(target)
                    xl.append(lm.x)
                    yl.append(lm.y)
                    templ.append((lm.x, lm.y))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return templ,labellist

landmarksT , landmarkstemp = [] , []
labelT , labeltemp = [] , []
for i in range(len(activities)):
    for file in os.listdir("Windows" + "/" + activities[i]):
        path = "Windows" + '/' + activities[i] + "/" + file
        arr = arr.removesuffix('.mp4')
        landmarkstemp , labeltemp = getTrainlandmarks(path,activities[i])
        landmarksT = landmarksT + landmarkstemp
        labelT = labelT + labeltemp

X_test , temp = getTrainlandmarks("finaltest.mp4",None)
X_train = landmarksT
y_train = labelT
listtargets = []
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
flagct = 0
# df = pd.DataFrame({
#     "prediction" : prediction
# })
# df.to_csv(r'my_data.csv', index=False)
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
