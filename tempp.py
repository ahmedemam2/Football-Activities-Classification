import cv2
import os
import shutil
import mediapipe as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

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
                if id == 12:
                    h, w, c = img.shape
                    # if target is None:
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
# for i in range(len(os.listdir("")))
# print(range(len(os.listdir("Windows" + "/" + arr))))
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
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (00, 185)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        # print(knn.predict(X_test))
        knn = prediction[ct]
        ct = ct+1
        text = knn
      # Using cv2.putText() method
        img = cv2.putText(img, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


