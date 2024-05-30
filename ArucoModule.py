import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    myList = os.listdir(path)
    augDict = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDict[key] = imgAug
    return augDict
   
def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    bboxes, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxes)
    return [bboxes, ids]

def augmentAruco(bbox, id, img, imgAug, drawID=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    h, w, c = imgAug.shape
    
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut += img
    
    if drawID:
        cv2.putText(imgOut, str(id), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)
        
    return imgOut

def main():
    cap = cv2.VideoCapture(0)
    augDict = loadAugImages("Markers")
    
    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDict.keys():
                    img = augmentAruco(bbox, id, img, augDict[int(id)])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
