from __future__ import division

# Author: Jean KOSSAIFI <jean.kossaifi@gmail.com>

import numpy as np
from numpy import rot90, sqrt
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..histogram import gradient, magnitude_orientation, compute_coefs, interpolate, normalise_histogram, interpolate_orientation


def test_gradient():
    """ tests for the gradient function 
    """
    # No duplication of the border
    image = np.array([[-1, 0, 1, 1],
                      [1, 0, 1, 1],
                      [1, 1, 1, 1],
                      [-1, 0, 1, 1],
                      [-1, 0, 1, 1]])
    gradx = np.array([[0, 1], [0, 0], [2, 1]])
    grady = np.array([[-1, 0], [0, 0], [1, 0]])
    resx, resy = gradient(image, same_size=False)
    assert_array_equal(gradx, resx)
    assert_array_equal(grady, resy)
    # With duplication of the border
    assert(np.shape(gradient(image, same_size=True)[0])==(5, 4))
    image = np.array([[1, 1],
                      [-1, 1]])
    gradx = np.array([[0, 0], [2, 2]])
    grady = np.array([[2, 0], [2, 0]])
    resx, resy = gradient(image, same_size=True)
    assert_array_equal(gradx, resx)
    assert_array_equal(grady, resy)


def test_magnitude_orientation():
    """ test for the magnitude_orientation function 
    """
    gx = np.array([[1, 0],
                  [0, 1]])
    gy = np.array([[0, 1],
                  [0, 1]])
    magnitude, orientation = magnitude_orientation(gx, gy)
    res_orientation = np.array([[0, 90], [0, 45]])
    res_magnitude = np.array([[1, 1], [0, sqrt(2)]])
    assert_array_equal(res_orientation, orientation)
    assert_array_equal(res_magnitude, magnitude)
    
    gx = np.array([[0, 1], [0, 1]])
    gy = np.array([[-1, 0], [1, 1]])
    magnitude, orientation = magnitude_orientation(gx, gy)
    assert(orientation[0, 0] == 270)
    assert(orientation[0, 1] == 0)
    assert(orientation[1, 0] == 90)
    assert(orientation[1, 1] == 45)


def test_compute_coefs():
    """ tests for the compute coefs function
    """
    csx, csy = (4, 4)
    dx = csx//2
    dy = csy//2
    n_cells_y, n_cells_x = (6, 4)
    coefs = compute_coefs(csy, csx, dy, dx, n_cells_y, n_cells_x)
    
    # Create an image to store the results
    res = np.tile(np.zeros((csx, csy)), (n_cells_x, n_cells_y))
    
    # We check that the sum of the coefficient for a given pixel is one
    res[dy:, dx:] += coefs[-(n_cells_x*csx - dx):, -(n_cells_y*csy - dy):]
    res[:-dy, dx:] += rot90(coefs[-(n_cells_y*csy - dy):, -(n_cells_x*csx - dx):])
    res[:-dy, :-dx] += rot90(rot90(coefs[-(n_cells_x*csx - dx):, -(n_cells_y*csy - dy):]))
    res[dy:, :-dx] += rot90(rot90(rot90(coefs[-(n_cells_y*csy - dy):, -(n_cells_x*csx - dx):])))
    
    assert np.all(res==1)


def test_interpolate_orientation():
    sy, sx = 4, 9
    nbins = 9
    orientation = np.reshape(np.arange(36)*10, (4, 9))
    res = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation=False)
    count = 0
    even = 0
    for i in res:
        for j in i:
            if even == 0:
                a = np.zeros(9)
                a[count] = 1
                assert_array_equal(j, a)
                even = 1
            else:
                a = np.zeros(9)
                a[count] = 0.5
                count += 1
                if count >= nbins:
                    count = 0
                a[count] = 0.5
                assert_array_equal(j, a)
                even = 0

    orientation = np.reshape(np.arange(36)*10, (4, 9))
    res = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation=True)
    assert_array_equal(res[0, 0, :], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert_array_equal(res[0, 1, :], np.array([0.75, 0.25, 0, 0, 0, 0, 0, 0, 0]))
    assert_array_equal(res[-1, -1, :], np.array([0.75, 0, 0, 0, 0, 0, 0, 0, 0.25]))


def test_interpolate():
    """ tests for the interpolate function
    """
    csx, csy = (4, 4)
    n_cells_x, n_cells_y = (3, 2)
    magnitude = np.zeros((csy*n_cells_y, csx*n_cells_x))
    magnitude[1, 1] = 1
    magnitude[3, 3] = 1
    magnitude[-1, -1] = 1
    magnitude[3, -1] = 1
    orientation = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0, 55],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15]])
    sy, sx = magnitude.shape
    hist = interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y)
    
    assert(np.sum(hist[:, :, 8])==1) # there is only one value in the bin 100 to 119 degree
    assert(hist[0, 0, 5]==1) #the 100 in the upper left corner is interpolated only to its own cell
    assert(hist[-1, -1, 0] == 0.25)
    assert(hist[-1, -1, 1] == 0.75) # interpolation only between the two bins
    assert(np.sum(hist[:, 2, 2] + hist[:, 2, 3]) == 1)


def test_normalise_histogram():
    """ tests for the normalise_histogram function
    """
    bx, by = (1, 1)
    n_cells_x, n_cells_y = (2, 2)
    nbins = 9
    
    orientation_histogram = np.array([[[0, 0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                       
                                       [[1, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0, 0]]])
    hist = normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins)
    assert_array_almost_equal(hist[0, 0], np.array([ 0, 0, 0, 0, 0, 0, 0, 1, 0]))
    assert_array_equal(hist[0, 1], np.zeros(nbins))
    assert_array_almost_equal(hist[1, 0], np.array([ 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]))
    assert_array_almost_equal(hist[1, 1], np.array([ 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]))
