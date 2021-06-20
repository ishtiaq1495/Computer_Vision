
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


def build_A(pts1, pts2):
    """
    Constructs the intermediate matrix A used in the total least squares 
    computation of an homography mapping pts1 to pts2.

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 2Nx9 matrix A that we'll use to solve for h
    """
    if pts1.shape != pts2.shape:
        raise ValueError('The source points for homography computation must have the same shape (%s vs %s)' % (
            str(pts1.shape), str(pts2.shape)))
    if pts1.shape[0] < 4:
        raise ValueError('There must be at least 4 pairs of correspondences.')
    num_pts = pts1.shape[0]
    # TODO: Create A which is 2N by 9...
    A = np.zeros((2*num_pts,9))
    #print(pts1)
    # TODO: iterate over the points and populate the rows of A.
    for i in range(2*num_pts):
        i_2=int(i/2)
        #print(i_2)
        if(i % 2 == 0):
            A[i][0]=pts1[i_2][0]
            A[i][1]=pts1[i_2][1]
            A[i][2]=1.0
            A[i][6]=pts1[i_2][0]*(-pts2[i_2][0])
            A[i][7]=pts1[i_2][1]*(-pts2[i_2][0])
            A[i][8]=-pts2[i_2][0]
        else:
            A[i][3]=pts1[i_2][0]
            A[i][4]=pts1[i_2][1]
            A[i][5]=1.0
            A[i][6]=pts1[i_2][0]*(-pts2[i_2][1])
            A[i][7]=pts1[i_2][1]*(-pts2[i_2][1])
            A[i][8]=-pts2[i_2][1]


    return A


def compute_H(pts1, pts2):
    """
    Computes an homography mapping one set of co-planar points (pts1)
    to another (pts2).

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 3x3 homography matrix that maps homogeneous coordinates of pts 1 to those in pts2.
    """
    # TODO: Construct the intermediate A matrix using build_A
    A = build_A(pts1,pts2)

    # TODO: Compute the symmetric matrix AtA.
    AtA =(A.transpose()).dot(A)

    # TODO: Compute the eigenvalues and eigenvectors of AtA.
    eig_vals, eig_vecs =np.linalg.eig(AtA) 

    # TODO: Determine which eigenvalue is the smallest
    min_eig_val_index =np.argmin(eig_vals)

    # TODO: Return the eigenvector corresponding to the smallest eigenvalue, reshaped
    # as a 3x3 matrix.
    min_eig_vec = eig_vecs[:,min_eig_val_index].reshape(3,3)

    return min_eig_vec


def bilinear_interp(image, point):
    """
    Looks up the pixel values in an image at a given point using bilinear
    interpolation. point is in the format (x, y).

    Args:
        image:      The image to sample
        point:      A tuple of floating point (x, y) values.

    Returns:
        A 3-dimensional numpy array representing the pixel value interpolated by "point".
    """
    # TODO: extract x and y from point
    y,x=point # see hw2_test, it is y-x not x-y it is because the sequence is height-width
    # TODO: Compute i,j as the integer parts of x, y
    i,j=int(x),int(y)
    # TODO: check that i + 1 and j + 1 are within range of the image. if not, just return the pixel at i, j
    if(image.shape[0]<=i+1) or (image.shape[1]<=j+1):
        return image[i][j]
    # TODO: Compute a and b as the floating point parts of x, y
    a,b=x-i,y-j
    # TODO: Take a linear combination of the four points weighted according to the inverse area around them
    # (i.e., the formula for bilinear interpolation)
    pixel_value = (1-a)*(1-b)*image[i][j]+a*(1-b)*image[i+1][j]+a*b*image[i+1][j+1]+(1-a)*b*image[i][j+1]
    return pixel_value

def apply_homography(H, points):
    """
    Applies the homography matrix H to the provided cartesian points and returns the results 
    as cartesian coordinates.

    Args:
        H:      A 3x3 floating point homography matrix.
        points: An Nx2 matrix of x,y points to apply the homography to.

    Returns:
        An Nx2 matrix of points that are the result of applying H to points.
    """

    # TODO: First, transform the points to homogenous coordinates by adding a `1`
    one_matrix = np.ones((points.shape[0],1))
    homogenous_points=np.hstack((points,one_matrix))

    # TODO: Apply the homography
    homography=H.dot(homogenous_points.transpose())
    # TODO: Convert the result back to cartesian coordinates and return the results
    point= (homography[:2]/homography[2]).transpose()
    return point

def warp_homography(source, target_shape, Hinv):
    """
    Warp the source image into the target coordinate frame using a provided
    inverse homography transformation.

    Args:
        source:         A 3-channel image represented as a numpy array.
        target_shape:   A 3-tuple indicating the desired results height, width, and channels, respectively
        Hinv:           A homography that maps locations in the result to locations in the source image.

    Returns:
        An image of target_shape with source's type containing the source image warped by the homography.
    """
    # TODO: allocation a numpy array of zeros that is size target_shape and the same type as source.
    result=np.zeros(target_shape,dtype=source.dtype)
    # TODO: Iterate over all pixels in the target image
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            # TODO: apply the homography to the x,y location
            point=apply_homography(Hinv,np.array([[i,j]]))[0]
            # TODO: check if the homography result is outside the source image. If so, move on to next pixel.
            p,q=point[0],point[1]
            if(p<0) or (p>source.shape[0]) or (q <0) or (q>source.shape[1]):
                pass
            # TODO: Otherwise, set the pixel at this location to the bilinear interpolation result.
            else:
                result[i][j]=bilinear_interp(source,(q,p))

    # return the output image
    return result


def rectify_image(image, source_points, target_points, crop):
    """
    Warps the input image source_points to the plane defined by target_points.

    Args:
        image:          The input image to warp.
        source_points:  The coordinates in the input image to warp from.
        target_points:  The coordinates to warp the corresponding source points to.
        crop:           If False, all pixels from the input image are shown. If true, the image is cropped to 
                        not show any black pixels.
    Returns:
        A new image containing the input image rectified to target_points.
    """

    # TODO: Compute the rectifying homography H that warps the source points to the
    # target points.
    H=compute_H(source_points[:,[1,0]],target_points[:,[1,0]])
    # TODO: Apply the homography to a rectangle of the bounding box of the of the image to find the
    # warped bounding box in the rectified space.
    source_bbox=np.zeros((4,2))
    source_bbox[1][0]=image.shape[0]
    source_bbox[2][1]=image.shape[1]
    source_bbox[3][0]=image.shape[0]
    source_bbox[3][1]=image.shape[1]
    target_bbox=apply_homography(H,source_bbox)
    # Find the min_x and min_y values in the warped space to keep.
    target_bbox_sorted_x=sorted(target_bbox[:,0])
    target_bbox_sorted_y=sorted(target_bbox[:,1])
    if crop:
        # TODO: pick the second smallest values of x and y in the warped bounding box
        x_min=target_bbox_sorted_x[1]
        y_min=target_bbox_sorted_y[1]
    else:
        # TODO: Compute the min x and min y of the warped bounding box
        x_min=target_bbox_sorted_x[0]
        y_min=target_bbox_sorted_y[0]

    # TODO: Compute a translation matrix T such that min_x and min_y will go to zero
    T=np.identity(3)
    T[0][2]=-x_min
    T[1][2]=-y_min
    # TODO: Compute the rectified bounding box by applying the translation matrix to
    # the warped bounding box.
    one_matrix = np.ones((target_bbox.shape[0],1))
    target_bbox=T.dot(np.hstack((target_bbox,one_matrix)).transpose())[:2].transpose()

    # TODO: Compute the inverse homography that maps the rectified bounding box to the original bounding box
    Hinv=compute_H(target_bbox,source_bbox)
    # Determine the shape of the output image
    target_bbox_sorted_x=sorted(target_bbox[:,0],reverse=True)
    target_bbox_sorted_y=sorted(target_bbox[:,1],reverse=True)
    if crop:
        # TODO: Determine the side of the final output image as the second highest X and Y values of the
        # rectified bounding box
        x_max=target_bbox_sorted_x[1]
        y_max=target_bbox_sorted_y[1]
    else:
        # TODO: Determine the side of the final output image as the maximum X and Y values of the
        # rectified bounding box
        x_max=target_bbox_sorted_x[0]
        y_max=target_bbox_sorted_y[0]

    # TODO: Finally call warp_homography to rectify the image and return the result
    new_image = warp_homography(image,(int(x_max),int(y_max),3),Hinv)
    return new_image

def blend_with_mask(source, target, mask):
    """
    Blends the source image with the target image according to the mask.
    Pixels with value "1" are source pixels, "0" are target pixels, and
    intermediate values are interpolated linearly between the two.

    Args:
        source:     The source image.
        target:     The target image.
        mask:       The mask to use

    Returns:
        A new image representing the linear combination of the mask (and it's inverse)
        with source and target, respectively.
    """

    # TODO: First, convert the mask image to be a floating point between 0 and 1
    # TODO: Next, use it to make a linear combination of the pixels
    new_image=np.zeros(source.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            grayscale=(float(mask[i][j][0])/255+float(mask[i][j][1])/255+float(mask[i][j][2])/255)/3
            new_image[i][j]=grayscale*source[i][j]+(1-grayscale)*target[i][j]

    # TODO: Convert the result to be the same type as source and return the result
    new_image=new_image.astype(source.dtype)
    return new_image

def composite_image(source, target, source_pts, target_pts, mask):
    """
    Composites a masked planar region of the source image onto a
    corresponding planar region of the target image via homography warping.
    
    Args:
        source:     The source image to warp onto the target.
        target:     The target image that the source image will be warped to.
        source_pts: The coordinates on the source image.
        target_pts: The corresponding coordinates on the target image.
        mask:       A greyscale image representing the mast to use.
    """
    # TODO: Compute the homography to warp points from the target to the source coordinate frame.
    H=compute_H(source_pts[:,[1,0]],target_pts[:,[1,0]])
    # TODO: Warp the source image to a new image (that has the same shape as target) using the homography.
    new_image=warp_homography(source,(target.shape[0],target.shape[1],3),np.linalg.inv(H))
    # TODO: Blend the warped images and return them.
    new_image=blend_with_mask(new_image,target,mask)
    return new_image

def rectify(args):
    """
    The 'main' function for the rectify command.
    """

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # load the destination points, or select some smart default ones if None
    if args.dst == None:
        height = np.abs(
            np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
        width = np.abs(
            np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
        args.dst = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    target_points = np.array(args.dst).reshape(4, 2)

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Compute the rectified image
    result = rectify_image(inputImage, source_points, target_points, args.crop)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


def composite(args):
    """
    The 'main' function for the composite command.
    """

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # load the target image
    logging.info('Loading target image %s' % (args.target))
    targetImage = load_image(args.target)

    # load the mask image
    logging.info('Loading mask image %s' % (args.mask))
    maskImage = load_image(args.mask)

    # If None, set the source points or sets them to the whole input image
    if args.source == None:
        (height, width, _) = inputImage.shape
        args.source = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # Loads the target points into a 4-by-2 array
    target_points = np.array(args.dst).reshape(4, 2)

    # Compute the composite image
    result = composite_image(inputImage, targetImage,
                             source_points, target_points, maskImage)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Warps an image by the computed homography between two rectangles.')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_rectify = subparsers.add_parser(
        'rectify', help='Rectifies an image such that the input rectangle is front-parallel.')
    parser_rectify.add_argument('input', type=str, help='The image to warp.')
    parser_rectify.add_argument('source', metavar='f', type=float, nargs=8,
                                help='A floating point value part of x1 y1 ... x4 y4')
    parser_rectify.add_argument(
        '--crop', help='If true, the result image is cropped.', action='store_true', default=False)
    parser_rectify.add_argument('--dst', metavar='x', type=float, nargs='+',
                                default=None, help='The four destination points in the output image.')
    parser_rectify.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_rectify.set_defaults(func=rectify)

    parser_composite = subparsers.add_parser(
        'composite', help='Warps the input image onto the target points of the target image.')
    parser_composite.add_argument(
        'input', type=str, help='The source image to warp.')
    parser_composite.add_argument(
        'target', type=str, help='The target image to warp to.')
    parser_composite.add_argument('dst', metavar='f', type=float, nargs=8,
                                  help='A floating point value part of x1 y1 ... x4 y4 defining the box on the target image.')
    parser_composite.add_argument(
        'mask', type=str, help='A mask image the same size as the target image.')
    parser_composite.add_argument('--source', metavar='x', type=float, nargs='+',
                                  default=None, help='The four source points in the input image. If ommited, the whole image is used.')
    parser_composite.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_composite.set_defaults(func=composite)

    args = parser.parse_args()
    args.func(args)
