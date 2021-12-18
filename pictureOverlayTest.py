import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def main(targetImageName, myVidPicName, videoName):
    # cap = cv2.VideoCapture(0)
    # imgTarget = cv2.imread("TargetImage.jpg")
    # myVid = cv2.VideoCapture('video.mp4')
    
    # cap = cv2.VideoCapture(videoName)
    imgWebcam = cv2.imread(videoName)
    imgTarget = cv2.imread(targetImageName)
    imgVideo = cv2.imread(myVidPicName)
    
    # detection = False
    # frameCounter = 0
    
    # success ,imgVideo = myVid.read()
    heigt, width, _ = imgTarget.shape
    imgVideo = cv2.resize(imgVideo, (width, heigt))
    
    plt.figure(figsize=(20, 20))
    plt.imshow(imgVideo)
    plt.title("imgVideo")
    plt.show()
    
    # cv2.imshow('target', imgTarget)
    # cv2.imshow('myVid', imgVideo)
    # cv2.waitKey(0)
    
    # orb = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures = 1000)
    knownKeypoints, knownDescription = orb.detectAndCompute(imgTarget, None)
    
    imgTarget = cv2.drawKeypoints(imgTarget, knownKeypoints, None)  
    
    plt.figure(figsize=(20, 20))
    plt.imshow(imgTarget)
    plt.title("imgTarget")
    plt.show()
    
    while True:
        # sucess, imgWebcam = cap.read()
        imgAug = imgWebcam.copy()
        keypoints2, description2 = orb.detectAndCompute(imgWebcam, None)
        imgWebcam = cv2.drawKeypoints(imgWebcam, keypoints2, None)  
        
        plt.figure(figsize=(20, 20))
        plt.imshow(imgWebcam)
        plt.title("imgWebcam")
        plt.show()
        
        # if not detection:
        #     myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     frameCounter = 0
        # else:
        #     if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
        #         myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #         frameCounter = 0
        #     sucess, imgVideo = myVid.read()
        #     imgVideo = cv2.resize(imgVideo, (width, heigt))
        
        bruteForce = cv2.BFMatcher()
        matches = bruteForce.knnMatch(knownDescription, description2, k=2)
        
        possible = []
        for m, n in matches:
            # if m.distance / n.distance < 1:
            if m.distance < 0.75 * n.distance:
                possible.append(m)
        
        imgFeatures = cv2.drawMatches(imgTarget, knownKeypoints, imgWebcam, keypoints2, possible, None, flags=2)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(imgFeatures)
        plt.title("imgFeatures")
        plt.show()
        
        possible = possible[:70]
        
        if len(possible) > 10:
            srcPoints = np.float32([knownKeypoints[m.queryIdx].pt for m in possible]).reshape(-1, 1, 2)
            dstPoints = np.float32([keypoints2[m.queryIdx].pt for m in possible]).reshape(-1, 1, 2)
            
            print(len(knownKeypoints))
            print(len(keypoints2))
            print(len(matches))
            print(len(possible))
            
            # srcPoints = np.array([knownKeypoints[m.queryIdx].pt for m in possible])
            # dstPoints = np.array([keypoints2[m.queryIdx].pt for m in possible])
            
            matrix, mask1 = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5)
            
            points = np.float32([[0, 0], [0, heigt], [width, heigt], [width, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(points, matrix)
            print(points)
            print(dst)
            
            img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
            
            plt.figure(figsize=(20, 20))
            plt.imshow(img2)
            plt.title("img2")
            plt.show()
            return None
            imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
            
            plt.figure(figsize=(20, 20))
            plt.imshow(imgWarp)
            plt.title("img2imgWarp")
            plt.show()
            
            mask = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
            cv2.fillPoly(mask, [np.int32(dst)], (255, 255, 255))
            maskInverse = cv2.bitwise_not(mask)
            imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInverse)
            
            plt.figure(figsize=(20, 20))
            plt.imshow(imgAug)
            plt.title("imgAug")
            plt.show()
            
            imgAug = cv2.bitwise_or(imgWarp, imgAug)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(imgAug)
        plt.title("resutl")
        plt.show()
        break
        # frameCounter += 1

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
        
    main(config["known_image"], config["target_image"], config["test_image"])