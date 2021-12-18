import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


class PictureOverlay:
    def __init__(self):
        self.__debug = True
        self.knownPicture = None
        self.knownPictureGray = None
        self.targetPicture = None
        self.targePictureGray = None
        self.targetVideo = None
        self.video = None
        
        self.__originalFrame = None
        self.__currentFrame = None
    
    def __getVideoCapture(self, fileName):
        """ return the video capture object if the object is already taken return None """
            
        capture = cv2.VideoCapture(fileName)
    
        if capture.isOpened() == False:
            print("video is opened")
            return None
        
        return capture
    
    def __getVideoWriter(self, sampleImage, outputFileName):
        """ return the video writer object """
        
        height, width, layers = sampleImage.shape
        videoWriter = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        return videoWriter
    
    def __preperImages(self, knownPictureFileName, targetPictureFileName):
        self.knownPicture = cv2.cvtColor(cv2.imread(knownPictureFileName), cv2.COLOR_BGR2RGB)
        self.targetPicture = cv2.cvtColor(cv2.imread(targetPictureFileName), cv2.COLOR_BGR2RGB)
        
        self.knownPictureGray = cv2.cvtColor(self.knownPicture, cv2.COLOR_RGB2GRAY)
        self.targetPictureGray = cv2.cvtColor(self.targetPicture, cv2.COLOR_RGB2GRAY)
        
        height, width, _ = self.knownPicture.shape
        self.targetPicture = cv2.resize(self.targetPicture, (width, height))
        self.targetPictureGray = cv2.resize(self.targetPictureGray, (width, height))
        
        # print(self.knownPicture.shape)
        # print(self.knownPictureGray.shape)
        # print(self.knownPicture.shape)
        # print(self.knownPicture.shape)
        
        self.__points = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)  ### 
    
    def __detectKeypoints(self):  # detectFeatures
        bruteForceMatcher = cv2.BFMatcher()
        self.featureExtractor = cv2.SIFT_create()
        # self.featureExtractor = cv2.ORB_create(nfeatures = 1000) 
        
        knownKeyPoints, knownDescription = self.featureExtractor.detectAndCompute(self.knownPictureGray, None)
        targetKeyPoints, targetDescription = self.featureExtractor.detectAndCompute(self.targetPictureGray, None)
        
        if self.__debug:
            test1 = cv2.drawKeypoints(self.knownPicture, knownKeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # test = cv2.drawKeypoints(self.targetPicture, targetKeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(test1)
            plt.title("keypoints")
            plt.show()
            
            test2 = cv2.drawKeypoints(self.targetPicture, targetKeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(test2)
            plt.title("keypoints")
            plt.show()
        
        matches = bruteForceMatcher.knnMatch(knownDescription, targetDescription, k = 2)
        # possibleMatches = list(filter(lambda match: match[0].distance / match[1].distance < 0.5, matches))
        
        # possibleMatches = [match for match in matches if match[0].distance / match[1].distance < 0.5]
        # possibleMatches_list = []
        # for m in matches:
        #     if m[0].distance / m[1].distance < 0.5:
        #         possibleMatches_list.append(m)
        # possibleMatches = np.asarray(possibleMatches_list)[:, 0]
        
        possibleMatches = []
        for m in matches:
            if m[0].distance / m[1].distance < 0.7:
            # if m[0].distance < 0.75 * m[].distance:
                possibleMatches.append(m)
        possibleMatches_arr = np.asarray(possibleMatches)[:, 0]
        
        
        if self.__debug:
            # imageMatches = cv2.drawMatchesKnn(self.knownPicture, knownDescription, self.targetPicture, targetDescription, possibleMatches[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # imageMatches = cv2.drawMatchesKnn(self.knownPicture, knownKeyPoints, self.targetPicture, targetKeyPoints, possibleMatches[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            imageMatches = cv2.drawMatchesKnn(self.knownPicture, knownKeyPoints, self.targetPicture, targetKeyPoints, possibleMatches[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(20, 20))
            plt.imshow(imageMatches)
            plt.title("keypoints matches")
            plt.show()
        
        # print(possibleMatches)
        # print(possibleMatches[0])
        # print(type(possibleMatches[0]))
        
        goodKnownKeypoints = np.array([knownKeyPoints[m.queryIdx].pt for m in possibleMatches_arr])
        goodTargetKeypoints = np.array([targetKeyPoints[m.trainIdx].pt for m in possibleMatches_arr])
        homographicMatrix, masked = cv2.findHomography(goodTargetKeypoints, goodKnownKeypoints, cv2.RANSAC, 5.0)
        
        if self.__debug:
            print(homographicMatrix)
        
        # wrappedPicture = cv2.warpPerspective(self.targetPicture, homographicMatrix, (self.__originalFrame.shape[1], self.__originalFrame.shape[0]))
                       # cv2.warpPerspective(rgb_r, H, (rgb_l.shape[1] + rgb_r.shape[1], rgb_l.shape[0]))
        wrappedPicture = cv2.warpPerspective(self.targetPicture, homographicMatrix, (self.knownPicture.shape[1] + self.targetPicture.shape[1], self.knownPicture.shape[0]))
        wrappedPicture[0:self.knownPicture.shape[0], 0:self.knownPicture.shape[1]] = self.knownPicture
        
        return wrappedPicture, homographicMatrix
    
    def __overlayImages(self, wrappedPicture, homographicMatrix):
        whiteColor = (255, 255, 255)
        dst = cv2.perspectiveTransform(self.__points, homographicMatrix)  ### 
        mask = np.zeros((self.__originalFrame.shape[0], self.__originalFrame.shape[1]), np.uint8)
        cv2.fillPoly(mask, [np.int32(dst)], whiteColor)
        maskInverse = cv2.bitwise_not(mask)
        self.__currentFrame = cv2.bitwise_and(self.__currentFrame, self.__currentFrame, mask = maskInverse)
        self.__originalFrame = cv2.bitwise_or(wrappedPicture, self.__currentFrame)
    
    def __overlay(self):
        wrappedPicture, homographicMatrix = self.__detectKeypoints()
        
        plt.figure(figsize=(20, 20))
        plt.imshow(wrappedPicture)
        plt.title("wrappedPicture")
        plt.show()
        
        # self.__overlayImages(wrappedPicture, homographicMatrix)
        # self.__showCurrentImage()        
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__originalFrame)  # __currentFrame
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def simpleOverlay(self, knownPictureFileName, targetPictureFileName, videoPictureFileName):
        self.__originalFrame = cv2.cvtColor(cv2.imread(videoPictureFileName), cv2.COLOR_BGR2RGB)
        self.__preperImages(knownPictureFileName, targetPictureFileName)
        self.__currentFrame = self.__originalFrame.copy()
        self.__overlay()
        cv2.waitKey(0)
    
    def overlayImage(self, knownPictureFileName, targetPictureFileName, videoFileName, outputFileName = "output.avi", videoOutput = True):
        """ overlay the target picture onto the known picture for each frame of the video, if videoOutput is true then there will be an output file of the video """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        
        self.__preperImages(knownPictureFileName, targetPictureFileName)
        
        if videoCapture is not None:
            while videoCapture.isOpened():
                ret, self.__originalFrame = videoCapture.read()
                
                if not ret or self.__quitDetected():
                    break
                
                if videoOutput and videoWriter is None:
                    videoWriter = self.__getVideoWriter(self.__originalFrame, outputFileName)
                
                self.__currentFrame = self.__originalFrame.copy()
                
                self.__overlay()
                
                if videoOutput:
                    videoWriter.write(self.__originalFrame)

            videoCapture.release()

            if videoOutput:
                videoWriter.release()
        cv2.destroyAllWindows()

    
    def overlayVideo(self, knownPictureFileName, targetVideoFileName, videoFileName, outputFileName = "output.avi", videoOutput = True):
        pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    # PictureOverlay().simpleOverlay(config["known_image"], config["target_image"], config["test_image"])
    # PictureOverlay().simpleOverlay(config["known_image"], "./Tests/left.jpg", config["test_image"])
    PictureOverlay().simpleOverlay("./Tests/left.jpg", "./Tests/right.jpg", config["test_image"])
    # PictureOverlay().overlayImage(config["known_image"], config["target_image"], config["test_video"], videoOutput = False)
    # PictureOverlay().overlayVideo(config["known_image"], config["target_video"], config["test_video"], videoOutput = False)
