"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1.

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')
    else:
        size = size / 2 # Dividing the size of the matrix to locate the middle of the matrix
        x, y = np.mgrid[-(np.floor(size)):(np.ceil(size)),
               -(np.floor(size)):(np.ceil(size))]  # Creating the matrix, such that middlee pixel is x = 0 and y = 0
        normal = 1.0 / (2.0 * np.pi * sigma ** 2.0)
        rv = np.exp(-((x ** 2.0 + y ** 2.0) / (2.0 * sigma ** 2.0))) * normal  # Applying the low pass filter
        rv = rv / rv.sum()  # Normalising the kernal, such that sum of all the values = 1
        rv = rv.astype('float32')  # Converting the kernal data type to float32
    return rv


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))

    # TODO: determine, using the kernel shape, the ith and jth locations to start at.
    image_height = img.shape[0] # image height
    image_weight = img.shape[1] # image weight

    kernal_height = kernel.shape[0] # kernel height
    kernal_weight = kernel.shape[1] # kernel weight

    offset_h = kernal_height // 2 # Creating offset
    offset_w = kernal_weight // 2 # Creating offset


    #Creating a zero matrix the same size as the input image
    c_v = np.zeros(img.shape)
    #Populating the zero matrix with only values of non edge locations of the input image matrix
    #This is done so that we can easily detect the edge locations
    c_v[offset_h:(image_height - offset_h), offset_w:(image_weight - offset_w)] = img[
                                                                                  offset_h:(image_height - offset_h),
                                                                                  offset_w:(image_weight - offset_w)]

    #Detecting the edge values of the input image by checking the positions that remain zeros in the zero matrix
    if c_v[i][j] == 0:
        summation = img[i][j]
    #Performs kernel convolution at locations i,j
    else:
        summation = 0
        for m in range(kernal_height):
            for n in range(kernal_weight):
                summation = summation + kernel[m][n] * img[i - offset_h + m][j - offset_w + n]
    return summation


def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    #Making a copy of the input image to save results
    image = img.copy()
    #Populating each pixel in the input by calling convolve_pixel and return results.
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x][y] = convolve_pixel(image, kernel, x, y)
    return image


def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    #The image is split into 3 channels, namely red, green and blue
    else:
        x, y, z = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return x, y, z


def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    #Merging the 3 channels after they have been convoled with the kernal
    new_vector = np.zeros((r.shape[0], r.shape[1], 3))
    r = np.array([r])
    g = np.array([g])
    b = np.array([b])
    new_vector[:, :, 0], new_vector[:, :, 1], new_vector[:, :, 2] = r, g, b
    new_vector = new_vector.astype('uint8')
    return new_vector


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Split it into three channels
    logging.info('Splitting it into 3 channels')
    (r, g, b) = split(inputImage)

    # compute the gaussian kernel
    logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                 (args.k, args.sigma))
    kernel = create_gaussian_kernel(args.k, args.sigma)

    # convolve it with each input channel
    logging.info('Convolving the first channel')
    r = convolve(r, kernel)
    logging.info('Convolving the second channel')
    g = convolve(g, kernel)
    logging.info('Convolving the third channel')
    b = convolve(b, kernel)

    # merge the channels back
    logging.info('Merging results')
    resultImage = merge(r, g, b)
    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
