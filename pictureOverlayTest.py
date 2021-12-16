import cv2
import json
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread("TargetImage.jpg")
myVid = cv2.VideoCapture('video.mp4')

detection = False
frameCounter = 0

success ,imgVideo = myVid.read()
heigt, width, _ = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (width, heigt))

orb = cv2.ORB_create(nfeatures = 1000)
knownKeypoints, knownDescription = orb.detectAndCompute(imgTarget, None)
# knownKeypoints, knownDescription = orb.detectAndCompute(imgTarget, None)

# imgTarget = cv2.drawKeypoints(imgTarget, knownKeypoints, None)  


while True:
    sucess, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    keypoints2, description2 = orb.detectAndCompute(imgWebcam, None)
    
    # imgTarget = cv2.drawKeypoints(imgTarget, keypoints2, None)  
    
    if not detection:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        sucess, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (width, heigt))
    
    bruteForce = cv2.BFMatcher()
    matches = bruteForce.knnMatch(knownDescription, description2, k = 2)
    
    possible = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            possible.append(m)
    
    imgFeatures = cv2.drawMatches(imgTarget, knownKeypoints, imgWebcam, keypoints2, possible, None, flag = 2)
    
    if len(possible) > 20:
        srcPoints = np.float32([knownKeypoints[m.queryIdx].pt for m in possible]).reshape(-1, 1, 2)
        dstPoints = np.float32([keypoints2[m.queryIdx].pt for m in possible]).reshape(-1, 1, 2)
        
        matrix, mask1 = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5)
        
        points = np.float32([[0, 0], [0, heigt], [width, heigt], [width, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(points, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
    
        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
        
        mask = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(mask, [np.int32(dst)], (255, 255, 255))
        maskInverse = cv2.bitwise_not(mask)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInverse)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)
    
    cv2.imshow("resutl", imgAug)
    cv2.waitKey(1)
    frameCounter += 1

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
