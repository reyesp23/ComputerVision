import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.
       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #take 4 corners
    #figure out their position aftter applying the transform
    #get image size
    height, width, _ = np.array(img).shape

    #get image corner positions
    top_left_corner = [0,0]
    top_right_corner = [0, width-1]
    bottom_left_corner = [height-1, 0]
    bottom_right_corner = [height-1, width-1]

    #transform corner positions by M
    top_left_corner_t = M @ [[top_left_corner[1]],[top_left_corner[0]],[1]]
    top_right_corner_t  = M @ [[top_right_corner[1]],[top_right_corner[0]],[1]]
    bottom_left_corner_t  = M @ [[bottom_left_corner[1]],[bottom_left_corner[0]],[1]]
    bottom_right_corner_t = M @ [[bottom_right_corner[1]],[bottom_right_corner[0]],[1]]

    #get lists with all x and y values present in the transformed corners
    #convert back from homogeneous coordinates

    x_values = [top_left_corner_t[0]/top_left_corner_t[2], 
                top_right_corner_t[0]/top_right_corner_t[2], 
                bottom_left_corner_t[0]/bottom_left_corner_t[2], 
                bottom_right_corner_t[0]/bottom_right_corner_t[2]]
    y_values = [top_left_corner_t[1]/top_left_corner_t[2], 
                top_right_corner_t[1]/top_right_corner_t[2], 
                bottom_left_corner_t[1]/bottom_left_corner_t[2], 
                bottom_right_corner_t[1]/bottom_right_corner_t[2]]

    #get min and max for x and y
    minY = np.min(y_values)
    maxY = np.max(y_values)
    minX = np.min(x_values)
    maxX = np.max(x_values)

    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    #calculate inverse transform (inverse warping)
    M_inv = np.linalg.inv(M)
    
    #bounds to iterate over the transformed image 
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    #looping over each pixel of the transformed image
    for x in range(minX, maxX):
        for y in range(minY, maxY):

            # inverse transform
            p_f = np.dot(M_inv,[[x],[y],[1]])

            # convert to cartesian coordinates
            x_f = p_f[0] / p_f[2]
            y_f = p_f[1] / p_f[2]

            #check if target pixel is out of bounds
            isOutOfBounds = (0 > x_f or x_f > img.shape[1] - 1) or (0 > y_f or y_f > img.shape[0] - 1)
            if not isOutOfBounds: 

                #get intensity from original image for each channel
                R = img[int(y_f), int(x_f), 0]
                G = img[int(y_f), int(x_f), 1]
                B = img[int(y_f), int(x_f), 2]
                
                #get feathering alpha
                if x < (minX + blendWidth):
                    alpha = float(x - minX)/blendWidth
                elif (maxX - blendWidth) < x: 
                    alpha = float(maxX - x)/blendWidth
                else:
                    alpha = 1.0

                #skip black pixels
                isBlackPixel = (R == 0 and G == 0 and B == 0)
                if not isBlackPixel:
                    #add to accumulator
                    acc[y,x,0] += (R * alpha)
                    acc[y,x,1] += (G * alpha)
                    acc[y,x,2] += (B * alpha) 
                    acc[y,x,3] += alpha


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    #channels 0-2 divided
    img = np.zeros((acc.shape[0], acc.shape[1], 4), dtype=np.uint8)

    for row in range(acc.shape[0]):
        for col in range(acc.shape[1]):
            for channel in range(3):

                n = acc[row,col,channel]
                d = acc[row,col,3]

                if d == 0: 
                    img[row, col, channel] = 0
                else: 
                    img[row,col,channel] = n/d

            img[row,col,3] = 1


    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.
       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)
         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        #get image bounding box
        minX_i, minY_i, maxX_i, maxY_i = imageBoundingBox(img, M)

        #update global bounding box if needed
        minX = np.min([minX, minX_i])
        maxX = np.max([maxX, maxX_i])
        minY = np.min([minY, minY_i])
        maxY = np.max([maxY, maxY_i])

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)

    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage