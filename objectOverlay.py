import os
import cv2
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mesh_renderer import MeshRenderer


class ObjectOverlay:
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
    
    def __preperImages(self, knownPictureFileName):
        """ read the knwon and target image and resize the target if needed for later use """
        
        self.__knownPicture = cv2.imread(knownPictureFileName)
        self.__knownPictureGray = cv2.cvtColor(self.__knownPicture, cv2.COLOR_RGB2GRAY)
    
    def __calibrateCamera(self, calibrationVideo, saveCalibration = False, savedFile = "camera-calibrate.pkl"):
        """ calibrate the camera and get the current camera matrix and meta information """
        
        if os.path.isfile(savedFile):
            rms, camera_matrix, dist_coefs, _rvecs, _tvecs = pickle.load(open(savedFile, 'rb'))
            return (rms, camera_matrix, dist_coefs, _rvecs, _tvecs)
        
        index = 0
        imagePoints = []
        objectPoints = []
        square_size = 2.88
        pattern_size = (9, 6)
        
        calibrationVideoCapture = self.__getVideoCapture(calibrationVideo)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        while calibrationVideoCapture.isOpened():
            sucess, picture = calibrationVideoCapture.read()
            
            if not sucess:
                break
            
            if index % 5 != 0:
                index += 1
                continue
            
            picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
            pictureGrey = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
            
            found, corners = cv2.findChessboardCorners(pictureGrey, pattern_size)
            
            if index == 0:
                height, width = picture.shape[:2]
            
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                corners = cv2.cornerSubPix(pictureGrey, corners, (5, 5), (-1, -1), term)
            else:
                continue
            
            if self.__debug:
                img_w_corners = cv2.drawChessboardCorners(picture, pattern_size, corners, found)
                cv2.imshow('Frame', picture)
                cv2.imshow("corner", img_w_corners)
                cv2.waitKey(1)

            imagePoints.append(corners.reshape(-1, 2))
            objectPoints.append(pattern_points)
            index += 1
        
        calibrationVideoCapture.release()
        cv2.destroyAllWindows()
        
        rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor = cv2.calibrateCamera(objectPoints, imagePoints, (width, height), None, None)
        
        if saveCalibration:
            pickle.dump(( rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor), open(savedFile, 'wb'))
        
        return (rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor)
    
    def __findFeatures(self, minimumDistance = 0.75):
        """ detect features within the known image and the video frame image and build the homographic matrix """
        
        self.__matcher = cv2.BFMatcher()
        self.__featureExtractor = cv2.SIFT_create()
        
        frameKeyPoints, frameDescription = self.__featureExtractor.detectAndCompute(self.__videoFrameGray, None)
        knownKeyPoints, knownDescription = self.__featureExtractor.detectAndCompute(self.__knownPictureGray, None)
        
        if self.__debug:
            test1 = cv2.drawKeypoints(self.__knownPicture, knownKeyPoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('keypoints', test1)
        
        matchPoints = list(self.__matcher.knnMatch(frameDescription, knownDescription, k = 2))
        matchPoints.sort(key=lambda x: x[0].distance / x[1].distance, reverse = False)
        matchPoints = [match for match in matchPoints if match[0].distance / match[1].distance < minimumDistance]
        
        if self.__debug:
            matchesImage = cv2.drawMatchesKnn(self.__videoFrame, frameKeyPoints, self.__knownPicture, knownKeyPoints, matchPoints, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matchesImage', matchesImage)
           
        try:
            matchPoints = np.asarray(matchPoints)[:, 0]
            goodFrameKeypoints = np.array([frameKeyPoints[m.queryIdx].pt for m in matchPoints])
            goodKnownKeypoints = np.array([knownKeyPoints[m.trainIdx].pt for m in matchPoints])
            homographicMatrix, masked = cv2.findHomography(goodKnownKeypoints, goodFrameKeypoints, cv2.RANSAC, 5.0)
            
        except Exception:
            print("could not find enught features, trying with larger minimum distance")
            return self.__findFeatures(minimumDistance + 0.25)
        
        return homographicMatrix, masked, goodFrameKeypoints, goodKnownKeypoints
    
    def __solveCameraPose(self, homographicMatrix, mask, cameraMatrix, distCoeffs, frameKeypoints, knownKeypoints, zoom = 100, offset = (0, 0)):
        """ get the rotation and translation vector of the camera using the solvePnP """
        
        imagePoints = np.array([frameKeypoints[i] for i, index in enumerate(mask) if index == 1])
        objectPoints = np.array([[knownKeypoints[i][0] / zoom - offset[0], knownKeypoints[i][1] / zoom - offset[1], 0] for i, index in enumerate(mask) if index == 1])
        
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags = 0)
        
        return rvec, tvec
    
    def __draw(self, img, imgpts):
        """ simple drawing of a 3d cube """
        
        blue = (0, 0, 255)
        green = (0, 255, 0)
        imgpts = np.int32(imgpts).reshape(-1, 2)
    
        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, green, -1)
    
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, blue, 3)
    
        return img
    
    def __detectAndRender(self, objectPath, calibrationVideo):
        """ detect the image in the frame and create a 3d model on it """
        
        (rms, camera_matrix, dist_coefs, rotationVecstor, translationVecstor) = self.__calibrateCamera(calibrationVideo, True)
        homographicMatrix, mask, frameKeypoints, knownKeypoints = self.__findFeatures()
        height, width, _ = self.__videoFrame.shape
        # self.__maskPicture(homographicMatrix, height, width)
        
        if self.__debug:
            r_vec, t_vec = self.__solveCameraPose(homographicMatrix, mask, camera_matrix, dist_coefs, frameKeypoints, knownKeypoints, zoom = 21)
            square_size = 2.88
            objectPoints = (
                17
                * square_size
                * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
            )
            
            imgpts = cv2.projectPoints(objectPoints, r_vec, t_vec, camera_matrix, dist_coefs)[0]
            self.__videoFrame = cv2.undistort(self.__videoFrame, camera_matrix, dist_coefs)
            self.__drawnImage = self.__draw(self.__videoFrame, imgpts)
        else:
            r_vec, t_vec = self.__solveCameraPose(homographicMatrix, mask, camera_matrix, dist_coefs, frameKeypoints, knownKeypoints, offset = (0, 5))
            self.__drawnImage = MeshRenderer(camera_matrix, width, height, objectPath).draw(self.__videoFrame,  r_vec, t_vec)
        
        self.__showCurrentImage()
    
    def __maskPicture(self, homographicMatrix, height, width):
        """ mask the underlying picture """
        whiteImage = np.ones((width, height), dtype="uint8") * 255
        whiteMask = cv2.warpPerspective(whiteImage, homographicMatrix, (width, height))
        whiteMask = cv2.bitwise_not(whiteMask)
        self.__videoFrame = cv2.bitwise_and(self.__videoFrame, self.__videoFrame, mask = whiteMask)
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__drawnImage)
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def render(self, knownPictureFileName, targetObjectFileName, videoFileName, calibrationVideo, outputFileName = "output.avi", videoOutput = True):
        """ render the 3d oject on the known image if found in the video """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        
        self.__knownPicture = cv2.imread(knownPictureFileName)
        self.__knownPictureGray = cv2.cvtColor(self.__knownPicture, cv2.COLOR_RGB2GRAY)
        
        if videoCapture is not None:
            while videoCapture.isOpened():
                ret, self.__videoFrame = videoCapture.read()
                
                if not ret or self.__quitDetected():
                    break
                
                if videoOutput and videoWriter is None:
                    videoWriter = self.__getVideoWriter(self.__videoFrame, outputFileName)
                
                self.__videoFrameGray = cv2.cvtColor(self.__videoFrame, cv2.COLOR_RGB2GRAY)
                
                self.__detectAndRender(targetObjectFileName, calibrationVideo)
                
                if videoOutput:
                    videoWriter.write(self.__videoFrame)

            videoCapture.release()

            if videoOutput:
                videoWriter.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    ObjectOverlay().render(config["known_image"], config["3d_object"], config["test_video"], "./Tests/calibration_vid.mp4", videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_dragon"], config["test_video"], "./Tests/calibration_vid.mp4", videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_Wood_House"], config["test_video"], "./Tests/calibration_vid.mp4", videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_Chess_Board"], config["test_video"], "./Tests/calibration_vid.mp4", videoOutput = False)
