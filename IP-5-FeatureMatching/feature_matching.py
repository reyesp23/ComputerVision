import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
from scipy import signal

import transformations

## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class HarrisKeypointDetector(KeypointDetector):

    def isLocalMaximum(self, y, x, harrisImage):
        window_size = 7
        height, width = harrisImage.shape
        # zero padding to include edge pixels
        padding = int((window_size - 1) / 2)
        harrisImage_padded = np.zeros((height + 2 * padding  , width + 2 * padding))
        harrisImage_padded[padding:-padding, padding:-padding] = harrisImage
        
        # corner strength matrix for the given window centered at y,x
        harrisImage_window = harrisImage_padded[y: y + window_size, x: x + window_size]

        # if the strength at the current point is max, return true
        return harrisImage_window.max() == harrisImage[y,x] 

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''

        harrisImage = np.zeros(srcImage.shape)
        orientationImage = np.zeros(srcImage.shape)
        window_size = 5
        sigma = 0.5
        offset = int((window_size - 1) / 2)
        
        # get derivatives using 3x3 Sobel operator
        Ix = ndimage.filters.sobel(srcImage, mode='reflect')
        Iy = ndimage.filters.sobel(srcImage, mode='reflect', axis=0)
        
        Iy2 = np.square(Iy)
        Ix2 = np.square(Ix)
        Ixy = Ix*Iy
        angles = np.arctan2(Iy,Ix)
    
        # apply gaussian filter
        Iy2_gauss = cv2.GaussianBlur(Iy2,(window_size,window_size),sigma, cv2.BORDER_REFLECT)
        Ix2_gauss = cv2.GaussianBlur(Ix2,(window_size,window_size),sigma, cv2.BORDER_REFLECT)
        Ixy_gauss = cv2.GaussianBlur(Ixy,(window_size,window_size),sigma, cv2.BORDER_REFLECT)

        #iterate on entire image
        height, width = srcImage.shape
        
        for y in range(height):
            for x in range(width):
                
                # get window start and end index, handle edge cases
                x_start = x - offset if x - offset < 0 else x
                y_start = y - offset if y - offset < 0 else y
                x_end = x + offset + 1 if x + offset > width else x + 1
                y_end = y + offset + 1 if y + offset > height else y + 1

                # get the H matrix coefficients for that window
                A = np.sum(Ix2_gauss[y_start: y_end, x_start: x_end])
                B = np.sum(Ixy_gauss[y_start: y_end, x_start: x_end])
                C = np.sum(Iy2_gauss[y_start: y_end, x_start: x_end])

                # create the H matrix from the entries in the gradient of the window
                H = np.array([[A,B],[B,C]])

                # use H to compute the corner strength function, c(H) at every pixel
                detH = np.linalg.det(H)
                traceH = np.matrix.trace(H)
                harrisScore = detH - 0.1 * (traceH**2)
                
                # store in a matrix with the same shape as the original image
                harrisImage[y,x] = harrisScore
        
        orientationImage = np.degrees(angles)
        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maximum in
                         its 7x7 neighborhood.
        '''
        height, width = harrisImage.shape

        destImage = np.zeros_like(harrisImage, np.bool)
        for y in range(height):
            for x in range(width):
                destImage[y,x] = self.isLocalMaximum(y, x, harrisImage)
        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue
                f = cv2.KeyPoint() #create a key point
                f.pt = (x, y)
                f.size = 10
                f.angle = orientationImage[y,x]
                f.response = harrisImage[y, x]
                features.append(f)
        return features

## Feature descriptors #########################################################

class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))
            x = int(f.pt[0])
            y = int(f.pt[1])

            T1 = [-x,-y,0]
            ang = np.radians(-f.angle)

            transformation_1 = transformations.get_trans_mx(np.array(T1)) #T1
            rotation = transformations.get_rot_mx(0,0,ang)
            scaling = transformations.get_scale_mx(.2,.2,0)
            transformation_2 = transformations.get_trans_mx(np.array([4,4,0])) #T2

            t = np.dot(transformation_2,np.dot(scaling,np.dot(rotation,transformation_1)))
            t1 = [[t[0][0], t[0][1], t[0][3]],[t[1][0], t[1][1], t[1][3]]]
            transMx = np.array(t1)
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            var = np.var(destImage)
            mean = np.mean(destImage)
            std = np.std(destImage)
            destImage = destImage - mean

            if var < 0.0000000001:
                destImage = np.zeros((windowSize,windowSize))
                destImage = destImage.flatten()
            else:
                destImage = destImage/std
                destImage = destImage.flatten()

            desc[i] = destImage
        return desc


## Feature matchers ############################################################

class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        distances = spatial.distance.cdist(desc1,desc2)
        
        # find indices for descriptor_2 matches that minimize the distance
        min_distance_desc2_indices = np.argmin(distances, axis= 1)
       
        # iterate on desc_1 
        for desc1_index in range(desc1.shape[0]):
            # find a desc2_index for each possible desc1_index
            desc2_index = min_distance_desc2_indices[desc1_index]
            
            # find the min distance given desc1_index and desc2_index
            distance = distances[desc1_index,desc2_index]
            
            # create a match
            m = cv2.DMatch(_queryIdx = desc1_index, _trainIdx = desc2_index, _distance = distance)
            
            # append to array
            matches.append(m)

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # find all possible distances between desc1 and desc2 and store in matrix form -> distances[desc1, desc2]
        distances = spatial.distance.cdist(desc1,desc2)
        
        # iterate on desc_1 
        for desc1_index in range(desc1.shape[0]):

            # find the two features with smallest distance to desc_1_i and store their indices
            desc2_indices = np.argpartition(distances[desc1_index], kth= 2)
            desc2_1_index = desc2_indices[0]
            desc2_2_index = desc2_indices[1]
            
            # compute ratio test
            distance1 = distances[desc1_index, desc2_1_index]
            distance2 = distances[desc1_index, desc2_2_index]
            ratio_distance = distance1/distance2
            
            # create a match
            m = cv2.DMatch(_queryIdx = desc1_index, _trainIdx = desc2_1_index, _distance = ratio_distance)
            
            # append to array
            matches.append(m)
        
        # TODO-BLOCK-END

        return matches
