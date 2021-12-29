import os
import cv2
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
# from mesh_renderer import MeshRenderer


class ObjectOverlay:
    def __init__(self):
        self.__debug = True
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
    
    def __calibrateCamera(self, calibrationVideo, videoFeed = False, saveCalibration = False, savedFile = "camera-calibrate.pkl"):
        """ calibrate the camera and get the current camera matrix and meta information """
        
        if os.path.isfile(savedFile):
            rms, camera_matrix, dist_coefs, _rvecs, _tvecs = pickle.load(open(savedFile, 'rb'))
            return rms, camera_matrix, dist_coefs, _rvecs, _tvecs
        
        index = 0
        imagePoints = []
        objectPoints = []
        square_size = 2.88
        pattern_size = (9, 6)
        
        calibrationVideoCapture = self.__getVideoCapture(calibrationVideo) if not videoFeed else calibrationVideo
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        while calibrationVideoCapture.isOpened():
            sucess, picture = calibrationVideoCapture.read()
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
            
            if self.__debug and index < 12:
                img_w_corners = cv2.drawChessboardCorners(picture, pattern_size, corners, found)
                plt.subplot(4, 3, index + 1)
                plt.imshow(img_w_corners)

            imagePoints.append(corners.reshape(-1, 2))
            objectPoints.append(pattern_points)
            index += 1
        
        rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor = cv2.calibrateCamera(objectPoints, imagePoints, (width, height), None, None)
        
        if saveCalibration:
            pickle.dump(( rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor), open(savedFile), 'wb')
        
        if self.__debug:
            print("\nRMS:", rms)
            print("camera matrix:\n", cameraMatrix)
            print("distortion coefficients: ", distCoeffs.ravel())
        
        return (rms, cameraMatrix, distCoeffs, rotationVecstor, translationVecstor)
    
    def __findFeatures(self, minimumDistance = 0.75):
        """ detect features within the known image and the video frame image and build the homographic matrix """
        
        self.__matcher = cv2.BFMatcher()
        self.__featureExtractor = cv2.SIFT_create()
        # self.__featureExtractor = cv2.ORB_create(nfeatures = 1000) 
        
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
        
        if self.__debug:
            print(homographicMatrix)
        
        return homographicMatrix, goodFrameKeypoints, goodKnownKeypoints
    
    def __solveCameraPose(self, homographicMatrix, cameraMatrix, distCoeffs, frameKeypoints, knownKeypoints):
        """ get the rotation and translation vector of the camera using the solvePnP """
        
        # possibility 1
        heigt, width, _ = self.__knownPicture.shape
        objectSrcPoints = np.float32([[0, 0], [0, heigt], [width, heigt], [width, 0]]).reshape(-1, 1, 2)
        # objectPoints = np.float32([[0, 0], [0, heigt - 1], [width - 1, heigt - 1], [width - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(objectSrcPoints, homographicMatrix)
        # objectPoints = np.array([[point[0][0], point[0][1], 0] for point in dst])
        objectPoints = np.array([[point[0][0], point[0][1], 0] for point in objectSrcPoints])
        
        # possibility 2
        ## objectPoints = [[point[0], point[1], -1] for point in frameKeypoints]
        
        retval, rvec, tvec = cv2.solvePnP(objectPoints, np.array(dst[:, 0, :]), cameraMatrix, distCoeffs, flags=0)
        
        return rvec, tvec
    
    def __draw(self, img, imgpts):
        """ """
        
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
        
        (rms, camera_matrix, dist_coefs, rotationVecstor, translationVecstor) = self.__calibrateCamera(calibrationVideo)
        self.__videoFrame = cv2.undistort(self.__videoFrame, camera_matrix, dist_coefs)  # needed ??
        homographicMatrix, frameKeypoints, knownKeypoints = self.__findFeatures()
        height, width = self.__videoFrame.shape[:2]
        
        r_vec, t_vec = self.__solveCameraPose(homographicMatrix, camera_matrix, dist_coefs, frameKeypoints, knownKeypoints) ## cv2.solvePnp
        
        if self.__debug:
            square_size = 60.5 #2.88
            objectPoints = (
                3
                * square_size
                * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
            )
            
            imgpts = cv2.projectPoints(objectPoints, r_vec, t_vec, camera_matrix, dist_coefs)[0]
            self.__videoFrame = cv2.undistort(self.__videoFrame, camera_matrix, dist_coefs)
            self.__drawnImage = self.__draw(self.__videoFrame, imgpts)
        else:
           # self.__drawnImage = MeshRenderer(camera_matrix, width, height, objectPath).draw(self.__videoFrame,  r_vec, t_vec)
            pass

        self.__showCurrentImage()
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__drawnImage)  # self.__videoFrame
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def render(self, knownPictureFileName, targetObjectFileName, videoFileName, objectPath, calibrationVideo, outputFileName = "output.avi", videoOutput = True):
        """ """
        
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
                
                self.__detectAndRender(objectPath, calibrationVideo)
                
                if videoOutput:
                    videoWriter.write(self.__videoFrame)

            videoCapture.release()

            if videoOutput:
                videoWriter.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
        
    ObjectOverlay().render(config["known_image"], config["3d_object"], config["test_video"], config["3d_object"], config["calibration_video"], videoOutput = False)
