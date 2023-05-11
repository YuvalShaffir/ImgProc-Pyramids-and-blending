import matplotlib.pyplot as plt
import numpy as np
import imageio
from scipy.ndimage.filters import convolve
from scipy import signal
from skimage.color import rgb2gray
import os

LOWEST_SHAPE_SIZE = 16

DEFAULT_KERNAL_ARRAY = np.array([1, 1])
DEFAULT_KERNAL_LENGTH = 2

GRAYSCALE = 1
RGB = 2


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # read the image - check if it is greyscale image:
    img = imageio.imread(filename)

    # greyscale representation
    if representation == GRAYSCALE:
        img_g = rgb2gray(img)
        img_g = img_g.astype('float64')
        return img_g

    # RGB representation
    if representation == RGB:
        img_rgb = img.astype('float64')
        img_rgb_norm = img_rgb / 255
        return img_rgb_norm


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    img = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(img, cmap="gray")
        plt.show()
    elif representation == RGB:
        plt.imshow(img)
        plt.show()


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurred_image = convolve(convolve(im, blur_filter, mode='constant'), blur_filter.T, mode='constant')
    # plt.imshow(blurred_image, cmap='gray')
    # plt.show()
    return blurred_image[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    blur_filter = blur_filter * 2

    up_scaled_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]), np.float64)
    up_scaled_im[::2, ::2] = im

    result = convolve(convolve(up_scaled_im, blur_filter, mode='constant'), blur_filter.T, mode='constant')

    return result


def blur_filter_generator(filter_size):
    """generates a blur filter vector from filter size"""
    kernel = DEFAULT_KERNAL_ARRAY
    vec_length = DEFAULT_KERNAL_LENGTH
    filter_vec = kernel

    while vec_length < filter_size:
        filter_vec = signal.convolve(filter_vec, kernel)
        vec_length += 1

    filter_vec = filter_vec / sum(filter_vec)  # normalization

    return filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = []
    filter_vec = blur_filter_generator(filter_size).reshape((1, filter_size))
    count = 0

    while count < max_levels:
        if im.shape[0] <= LOWEST_SHAPE_SIZE or im.shape[1] <= LOWEST_SHAPE_SIZE:
            break
        pyr.append(im)
        im = reduce(im, filter_vec)
        count += 1

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    lpyr = []
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    for lvl in range(len(pyr)-1):
        lpyr.append(pyr[lvl] - expand(pyr[lvl+1], filter_vec))
    lpyr.append(pyr[len(pyr)-1])
    # for lvl in range(len(lpyr)):
    #     plt.imshow(lpyr[lvl], cmap='gray')
    #     plt.show()
    return lpyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    # L_n + L_(n-1) = G_(n-1)
    # if we sum all the levels of the laplacian we get the original image
    index = len(lpyr) - 1
    tmp_pyr = lpyr

    for lvl in range(len(coeff)):
        lpyr[lvl] = coeff[lvl] * lpyr[lvl]

    while index > 0:
        tmp_pyr[index - 1] = lpyr[index - 1] + expand(tmp_pyr[index], filter_vec)
        index -= 1

    img = tmp_pyr[0]
    return img


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    # todo: strech the values to [0,1]
    pyr_width = [0]
    pyr_height = pyr[0].shape[0]
    rows = []
    columns = []

    for lvl in range(levels):
        pyr_width.append(pyr[lvl].shape[1])
        rows.append(pyr[lvl].shape[0])
        columns.append([sum(pyr_width[:-1]), sum(pyr_width)])

    large_gaussian_pyr = np.zeros((pyr_height, sum(pyr_width)), np.float64)
    for lvl in range(levels):
        large_gaussian_pyr[0:rows[lvl]:, columns[lvl][0]:columns[lvl][1]:] = pyr[lvl]

    return large_gaussian_pyr


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    result = render_pyramid(pyr, levels)
    plt.imshow(result)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    lpyr1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lpyr2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_pyr, filter_vec3 = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    lpyr3 = []
    for lvl in range(len(lpyr1)):
        lpyr3.append(mask_pyr[lvl] * lpyr1[lvl] + (1 - mask_pyr[lvl]) * (lpyr2[lvl]))

    index = len(lpyr3) - 1
    tmp_pyr = lpyr3
    while index > 0:
        tmp_pyr[index - 1] = lpyr3[index - 1] + expand(tmp_pyr[index], filter_vec1)
        index -= 1

    img = tmp_pyr[0]

    return img


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    max_levels = 10
    filter_size_im = 5
    filter_size_mask = 3
    bob_ross_im = read_image(relpath('externals/bob ross.jpg'), 2)
    shmuel_im = read_image(relpath('externals/shmuel.jpg'), 2)
    mask = np.round(read_image(relpath('externals/bob ross mask.jpg'), 1)).astype(np.bool)

    red = pyramid_blending(shmuel_im[:, :, 0], bob_ross_im[:, :, 0], mask.astype(np.float64), max_levels, filter_size_im, filter_size_mask)
    green = pyramid_blending(shmuel_im[:, :, 1], bob_ross_im[:, :, 1], mask.astype(np.float64), max_levels, filter_size_im, filter_size_mask)
    blue = pyramid_blending(shmuel_im[:, :, 2], bob_ross_im[:, :, 2], mask.astype(np.float64), max_levels, filter_size_im, filter_size_mask)
    blended_im = np.dstack((red, green, blue))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(bob_ross_im)
    axs[0, 1].imshow(shmuel_im)
    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 1].imshow(blended_im)
    plt.show()

    return bob_ross_im, shmuel_im, mask, blended_im


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    max_levels = 10
    filter_size_im = 5
    filter_size_mask = 3
    harry_potter_im = read_image(relpath('externals/harry potter.jpg'), 2)
    avital_im = read_image(relpath('externals/avital.jpg'), 2)
    mask = np.round(read_image(relpath('externals/harry potter mask.jpg'), 1)).astype(np.bool)

    red = pyramid_blending(avital_im[:, :, 0], harry_potter_im[:, :, 0], mask.astype(np.float64), max_levels, filter_size_im, filter_size_mask)
    green = pyramid_blending(avital_im[:, :, 1], harry_potter_im[:, :, 1], mask.astype(np.float64), max_levels, filter_size_im,
                             filter_size_mask)
    blue = pyramid_blending(avital_im[:, :, 2], harry_potter_im[:, :, 2], mask.astype(np.float64), max_levels, filter_size_im,
                            filter_size_mask)
    blended_im = np.dstack((red, green, blue))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(harry_potter_im)
    axs[0, 1].imshow(avital_im)
    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 1].imshow(blended_im)
    plt.show()

    return harry_potter_im, avital_im, mask, blended_im
