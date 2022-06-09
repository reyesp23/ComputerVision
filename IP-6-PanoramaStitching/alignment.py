import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        row1 = (i*2)
        row2 = (i*2) + 1

        A[row1,0] = a_x
        A[row1,1] = a_y
        A[row1,2] = 1
        A[row1,3] = 0
        A[row1,4] = 0
        A[row1,5] = 0
        A[row1,6] = -a_x*b_x
        A[row1,7] = -a_y*b_x
        A[row1,8] = -b_x

        A[row2,0] = 0
        A[row2,1] = 0
        A[row2,2] = 0
        A[row2,3] = a_x
        A[row2,4] = a_y
        A[row2,5] = 1
        A[row2,6] = -a_x*b_y
        A[row2,7] = -a_y*b_y
        A[row2,8] = -b_y

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #Homography to be calculated
    H = np.eye(3)
    H = Vt[-1].reshape(3,3)
    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    max_inlier_indices = []

    #for each RANSAC iteration
    for n in range(nRANSAC):
        if m == eTranslate:
            # get a random sample
            sample_match = np.random.choice(matches, 1)[0]
            
            #get keypoint on the first image
            id1 = sample_match.queryIdx
            p1 = f1[id1].pt

            #get keypoint on the second image
            id2 = sample_match.trainIdx
            p2 = f2[id2].pt
            
            #get ranslation on each axis
            u = p2[0] - p1[0]
            v = p2[1] - p1[1]
            
            #build translation matrix
            T = np.eye(3)
            T[0,2] = u
            T[1,2] = v
            
            #get inliers to the translation matrix
            inlier_indices = getInliers(f1, f2, matches, T, RANSACthresh)
            
            #count inliers and update max_inliers
            if len(inlier_indices) > len(max_inlier_indices):
                max_inlier_indices = inlier_indices
            
        elif m == eHomography:
            sample_matches = np.random.choice(matches, 4)
            H = computeHomography(f1, f2, sample_matches)
            inlier_indices = getInliers(f1, f2, matches, H, RANSACthresh)
            
            if len(inlier_indices) > len(max_inlier_indices):
                max_inlier_indices = inlier_indices
    
    #compute least squares for max inliers
    M = leastSquaresFit(f1, f2, matches, m, max_inlier_indices)
 
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
               
        #get keypoint on the first image
        id1 = matches[i].queryIdx
        p1 = f1[id1].pt
        p1_h = [[p1[0]],[p1[1]],[1]]
        
        #get keypoint on the second image
        id2 = matches[i].trainIdx
        p2 = f2[id2].pt
        p2_h = [[p2[0]],[p2[1]],[1]]  
        
        #trasform point from image 1 using M
        p1_t_h = M @ p1_h

        #convert back from homogeneous coordinates
        p1_t = (p1_t_h[0]/p1_t_h[2], p1_t_h[1]/p1_t_h[2])
        
        #get the distance between the transformed point and the actual match
        d = (((p1_t[0] - p2[0] )**2) + ((p1_t[1]-p2[1])**2) )**0.5
        
        #if distance is within the threshold, append the match index
        if d <= RANSACthresh:
            inlier_indices.append(i)

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    M = np.eye(3)

    if m == eTranslate:

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
        
            inliers = np.array(matches)[inlier_indices]
           
            #get keypoint on the first image
            id1 = inliers[i].queryIdx
            p1 = f1[id1].pt

            #get keypoint on the second image
            id2 = inliers[i].trainIdx
            p2 = f2[id2].pt
            
            u += p2[0] - p1[0]
            v += p2[1] - p1[1]

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        inliers = np.array(matches)[inlier_indices]
        M = computeHomography(f1, f2, inliers)

    else:
        raise Exception("Error: Invalid motion model.")

    return M

