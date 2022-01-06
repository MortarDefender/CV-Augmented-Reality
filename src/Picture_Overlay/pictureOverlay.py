import cv2
import json
import numpy as np


class PictureOverlay:
    def __init__(self):
        self.__debug = False
        self.__knownPicture = None
        self.__targetPicture = None
        self.__knownPictureGray = None
        
        self.__videoFrame = None
        self.__videoFrameGray = None
    
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
        """ read the knwon and target image and resize the target if needed for later use """
        
        self.__knownPicture = cv2.imread(knownPictureFileName)
        self.__knownPictureGray = cv2.cvtColor(self.__knownPicture, cv2.COLOR_RGB2GRAY)
        
        height, width, channels = self.__knownPicture.shape
        self.__targetPicture = cv2.imread(targetPictureFileName)
        self.__targetPicture = cv2.resize(self.__targetPicture, (width, height))
    
    def __preperAndGetFrame(self, targetVideoCapture):
        """ read a picture from the targetVideoCapture if the video has ended loop it again """
        
        height, width, channels = self.__knownPicture.shape
        sucess, self.__targetPicture = targetVideoCapture.read()
        
        if not sucess:
            targetVideoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            sucess, self.__targetPicture = targetVideoCapture.read()
        
        self.__targetPicture = cv2.resize(self.__targetPicture, (width, height))
    
    def __detectFeatures(self, minimumDistance = 1):
        """ detect features within the known image and the video frame image and build the homographic matrix """
        
        self.__matcher = cv2.BFMatcher()
        self.__featureExtractor = cv2.SIFT_create()
        # self.__featureExtractor = cv2.ORB_create(nfeatures = 1000) 
        
        frameKeyPoints, frameDescription = self.__featureExtractor.detectAndCompute(self.__videoFrameGray, None)
        knownKeyPoints, knownDescription = self.__featureExtractor.detectAndCompute(self.__knownPictureGray, None)
        
        if self.__debug:
            test1 = cv2.drawKeypoints(self.__knownPicture, knownKeyPoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('keypoints', test1)
        
        matches = list(self.__matcher.knnMatch(frameDescription, knownDescription, k = 2))
        matches.sort(key=lambda x: x[0].distance / x[1].distance, reverse = False)
        matches = [match for match in matches if match[0].distance / match[1].distance < minimumDistance][:30]
        
        if self.__debug:
            matchesImage = cv2.drawMatchesKnn(self.__videoFrame, frameKeyPoints, self.__knownPicture, knownKeyPoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matchesImage', matchesImage)
           
        try:
            matches = np.asarray(matches)[:, 0]
            goodFrameKeypoints = np.array([frameKeyPoints[m.queryIdx].pt for m in matches])
            goodKnownKeypoints = np.array([knownKeyPoints[m.trainIdx].pt for m in matches])
            homographicMatrix, masked = cv2.findHomography(goodKnownKeypoints, goodFrameKeypoints, cv2.RANSAC, 5.0)
            
        except Exception:
            print("could not find enught features, trying with larger minimum distance")
            return self.__detectFeatures(minimumDistance + 0.25)
        
        if self.__debug:
            print(homographicMatrix)
        
        return homographicMatrix
    
    def __overlayImages(self, homographicMatrix):
        """ replace the detected image with the target image using the homographic matrix """
        
        height, width, channels = self.__videoFrame.shape
        wrappedPicture = cv2.warpPerspective(self.__targetPicture, homographicMatrix, (width, height))
        whiteImage = np.ones((self.__targetPicture.shape[0], self.__targetPicture.shape[1]), dtype="uint8") * 255
        whiteMask = cv2.warpPerspective(whiteImage, homographicMatrix, (width, height))
        whiteMask = cv2.bitwise_not(whiteMask)
        self.__videoFrame = cv2.bitwise_and(self.__videoFrame, self.__videoFrame, mask = whiteMask)
        self.__videoFrame += wrappedPicture
    
    def __detectAndOverlay(self):
        """ detect keypoints in the two pictures, find the homographic Matrix and replace the knwon image with the target image """
        
        homographicMatrix = self.__detectFeatures()
        self.__overlayImages(homographicMatrix)
        self.__showCurrentImage()        
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__videoFrame)
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def __runOverlayLoop(self, videoWriter, videoCapture, outputFileName = "output.avi", videoOutput = True, middleRunFunction = None):
        """ main overlay loop, run over the videoCapture object until the video is done  """
        
        if videoCapture is not None:
            while videoCapture.isOpened():
                ret, self.__videoFrame = videoCapture.read()
                
                if not ret or self.__quitDetected():
                    break
                
                if videoOutput and videoWriter is None:
                    videoWriter = self.__getVideoWriter(self.__videoFrame, outputFileName)
                
                if middleRunFunction is not None:
                    middleRunFunction()
                
                self.__videoFrameGray = cv2.cvtColor(self.__videoFrame, cv2.COLOR_RGB2GRAY)
                
                self.__detectAndOverlay()
                
                if videoOutput:
                    videoWriter.write(self.__videoFrame)

            videoCapture.release()

            if videoOutput:
                videoWriter.release()
        cv2.destroyAllWindows()
        
    
    def overlayImage(self, knownPictureFileName, targetPictureFileName, videoFileName, outputFileName = "output.avi", videoOutput = True):
        """ overlay the target picture onto the known picture for each frame of the video, if videoOutput is true then there will be an output file of the video """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        
        self.__preperImages(knownPictureFileName, targetPictureFileName)
        
        self.__runOverlayLoop(videoWriter, videoCapture, outputFileName, videoOutput)

    
    def overlayVideo(self, knownPictureFileName, targetVideoFileName, videoFileName, outputFileName = "output.avi", videoOutput = True):
        """ overlay the target video onto the known picture for each frame of the video, if videoOutput is true then there will be an output file of the video """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        targetVideoCapture = self.__getVideoCapture(targetVideoFileName)
        
        self.__knownPicture = cv2.imread(knownPictureFileName)
        self.__knownPictureGray = cv2.cvtColor(self.__knownPicture, cv2.COLOR_RGB2GRAY)
        
        self.__runOverlayLoop(videoWriter, videoCapture, outputFileName, videoOutput, lambda : self.__preperAndGetFrame(targetVideoCapture))
        
        targetVideoCapture.release()

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    PictureOverlay().overlayImage(config["known_image"], config["target_image"], config["test_video"], videoOutput = False)
    # PictureOverlay().overlayVideo(config["known_image"], config["target_video"], config["test_video"], videoOutput = False)
