import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


class PictureOverlay:
    def __init__(self):
        self.__debug = False
        self.knownPicture = None
        self.knownPictureGray = None
        self.targePicture = None
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
        self.knownPicture = cv2.imread(knownPictureFileName, cv2.COLOR_BGR2RGB)
        self.targePicture = cv2.imread(targetPictureFileName, cv2.COLOR_BGR2RGB)
        self.knownPictureGray = cv2.cvtColor(self.knownPicture, cv2.COLOR_RGB2GRAY)
        self.targePictureGray = cv2.cvtColor(self.targePicture, cv2.COLOR_RGB2GRAY)
        
        height, width, _ = self.knwonPicture.shape
        self.targePicture = cv2.resize(self.targePicture, (height, width))
    
    def __detectKeypoints(self):  # detectFeatures
        bruteForceMatcher = cv2.BFMatcher()
        featureExtractor = cv2.SIFT_create()
        
        knownKeyPoints, knownDescription = featureExtractor.detectAndCompare(self.knwonPictureGray, None)
        targetKeyPoints, targetDescription = featureExtractor.detectAndCompare(self.targetPictureGray, None)
        
        if self.__debug:
            test = cv2.drawKeypoints(self.knownPicture, knownKeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # test = cv2.drawKeypoints(self.targePicture, targetKeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(test)
            plt.title("keypoints")
            plt.show()
        
        matches = bruteForceMatcher.knnMatch(knownDescription, targetDescription, k = 2)
        possibleMatches = list(filter(lambda match: match[0].disctance / match[1].disctance < 0.5, matches))
        
        # possibleMatches = [match for match in matches if match[0].disctance / match[1].disctance < 0.5]
        
        if self.__debug:
            imageMatches = cv2.drawMatchesKnn(self.knownPicture, knownDescription, self.targePicture, targetDescription, possibleMatches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(20, 20))
            plt.inshow(imageMatches)
            plt.title("keypoints matches")
            plt.show()
        
        goodKnownKeypoints = np.array([knownKeyPoints[match.queryIdx].pt for match in possibleMatches])
        goodTargetKeypoints = np.array([targetKeyPoints[match.queryIdx].pt for match in possibleMatches])
        homographicMatrix, masked = cv2.findHomography(goodKnownKeypoints, goodTargetKeypoints, cv2.RANSAC, 5.0)
        
        if self.__debug:
            print(homographicMatrix)
        
        wrappedPicture = cv2.warpPerspective(self.targePicture, homographicMatrix, (self.origPicture.shape[1], self.origPicture.shape[0]))
    
    def __overlayImages(self):
        mask = np.zeros((self.origPicture.shape[0], self.origPicture[1]), np.uint8)
        # cv2.fillPoly()
    
    def __overlay(self):
        pass
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__originalFrame)
    
    def overlayImage(self, knownPictureFileName, targetPictureFileName, videoFileName, outputFileName = "output.avi", videoOutput = True):
        """ overlay the target picture onto the known picture for each frame of the video, if videoOutput is true then there will be an output file of the video """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        
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
    
    PictureOverlay().overlayImage(config["known_image"], config["target_image"], config["test_video"], videoOutput = False)
    # PictureOverlay().overlayVideo(config["known_image"], config["target_video"], config["test_video"], videoOutput = False)

# # ======= imports
# pass

# # ======= constants
# pass

# # === template image keypoint and descriptors
# pass

# # ===== video input, output and metadata
# pass

# # ========== run on all frames
# while True:
#     # ====== find keypoints matches of frame and template
#     # we saw this in the SIFT notebook
#     pass

#     # ======== find homography
#     # also in SIFT notebook
#     pass

#     # ++++++++ do warping of another image on template image
#     # we saw this in SIFT notebook
#     pass

#     # =========== plot and save frame
#     pass

# # ======== end all
# pass
