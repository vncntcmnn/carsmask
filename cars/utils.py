import cv2
import skimage
import numpy as np
from scipy import ndimage
from functools import reduce
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.color import rgb2gray
from scipy.ndimage.morphology import binary_opening
from skimage.segmentation import felzenszwalb, find_boundaries


def remove_too_small_shape(mask, shape_prod=100):
    """Remove labels from a mask if those are too small.
    Get cell for each label and if the product of the shape is less
    than the shape_prod value remove it.

    Args:
        mask (numpy.ndarray): Mask numpy.ndarray (2D) of 0 and 1.
        shape_prod (int, optional): Shape product threshold. Defaults to 100.

    Returns:
        numpy.ndarray: 2D processed mask, the input without the too small labels.
    """
    l, c = ndimage.label(mask)
    for i in range(c):
        obj_indices = ndimage.find_objects(l)[i]
        cell = mask[obj_indices]
        if np.product(cell.shape) < shape_prod:
            mask[obj_indices] = 0
    return mask


def remove_too_small_pix(mask, pixels=10):
    """Remove labels from a mask if those are too small.
    Get cell for each label and if the number of 1s is less
    than the pixels value remove it.

    Args:
        mask (numpy.ndarray): Mask numpy.ndarray (2D) of 0 and 1.
        pix (int, optional): Pixels number threshold. Defaults to 10.

    Returns:
        numpy.ndarray: 2D processed mask, the input without the too small labels.
    """
    l, c = ndimage.label(mask)
    for i in range(c):
        if np.sum(l == i + 1) < pixels:
            mask[l == i + 1] = 0
    return mask


def find_labels_to_process(mask, shape_prod=400, pix=None):
    """Find labels who are supposed to contain more than a single car.
    Get cell for each label and if the product of the shape (respectively
    number of pixels/values greater than 0) is more than the shape_prod (respectively pix)
    value consider it as to process. If both shape_prod and pix have a value, pix is used.

    Args:
        mask (numpy.ndarray): Mask numpy.ndarray (2D) of 0 and 1.
        shape_prod (int, optional): Shape product threshold. Defaults to 400.
        pix(int, optional): Number of pixels as threshold. Defaults to None.

    Returns:
        List: List of labels too big to contain a single car.
    """
    toproc = []
    l, _ = ndimage.label(mask)
    for label_ind, label_coords in enumerate(ndimage.find_objects(l)):
        cell = mask[label_coords]
        # Check if the label size is too small
        if pix is not None:
            if np.sum(cell > 0) > pix:
                toproc.append(label_ind)
        elif shape_prod is not None:
            if np.product(cell.shape) > shape_prod:
                toproc.append(label_ind)
        else:
            print("shape_prod or pix value is needed as threshold.")
            break
    return toproc


def find_cars_orientation(img, mask, plot=False):
    """Find cars orientation in a colored cell/image by finding the min.
    rectangle area around it.
    Remove the background of the image using the mask. Then search for the min.
    area rectangle that can be drawn around it. As cars are usually parked next to each
    other and not one behind another, this can be used to find the cars orientation.

    Args:
        img (numpy.ndarray): Colored image cell to process (3 channels), supposed to contain
        more than a car.
        mask (numpy.ndarray): Mask numpy.ndarray (2D) of 0 and 1.

    Returns:
        float: Angle of the rectangle drawn around the cars (i.e. supposed orientation of
        the cars).
    """
    # Load image as HSV and select saturation
    hh, ww, cc = img.shape
    # Remove background for the colored image
    for i in range(cc):
        img[:, :, i][mask == 0] = 0

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    ret, thresh = cv2.threshold(gray, 0, 255, 0)

    # Find outer contour
    cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Get rotated rectangle from outer contour
    rotrect = cv2.minAreaRect(cntrs[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    # Draw rotated rectangle on copy of img as result
    result = img.copy()
    # cv2.drawContours(result,[box],0,(0,0,255),2)
    cv2.fillPoly(result, [box], (0, 0, 255))

    # Get angle from rotated rectangle
    angle = rotrect[-1]

    # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # Otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    if plot:
        cv2.imshow("THRESH", thresh)
        cv2.imshow("RESULT", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return angle, result, box


def generate_car_mask(angle):
    """Generate a car mask from the input angle.
    This mask will be used to go through a mask containing more than a single car
    and find the areas overlapping.

    Args:
        angle (float): Orientation angle of the car.

    Returns:
        numpy.ndarray: 2D minimum mask to contain the oriented car mask.
    """
    # TODO: Need to take into account two cars parked in behing another.
    car_rect = np.zeros((100, 100))

    rr, cc = skimage.draw.rectangle((40, 40), end=(54, 46))
    car_rect[rr, cc] = 1

    car_rect_rot = skimage.transform.rotate(car_rect, angle=angle + 90)
    car_rect_rot[car_rect_rot > 0] = 1

    # find min box around it
    # TODO: Need to find a better way
    min_x = 999
    min_y = 999
    max_x = 0
    max_y = 0
    for i in range(car_rect_rot.shape[0]):
        for j in range(car_rect_rot.shape[0]):
            if car_rect_rot[i, j] == 1:
                if i < min_x:
                    min_x = i
                if i > max_x:
                    max_x = i
                if j < min_y:
                    min_y = j
                if j > max_y:
                    max_y = j

    car_selected = car_rect_rot[min_x:max_x, min_y:max_y]
    return car_selected


def open_remove(img, iters=3):
    """Aplly binary open with iters iterations on a mask. Remove small labels
    with less than 30 (those cannot contain a car). Then remove small labels with less than 160
    pixels which could contain a single car and store them in a list (returned).

    Args:
        img (numpy.ndarray): Image, without background, to process.
        iters (int, optional): Number of iterations for binary_opening. Defaults to 3.

    Returns:
        (numpy.ndarray, list(numpy.ndarray)): Return a tuple of two objects: mask to keep
        processing and removed masks/labels which are supposed to contain a single car each.
    """
    igbb = binary_opening(
        img, iterations=iters, structure=morph.square(2)
    )  # previously square 3
    mask = igbb.copy()
    mask[mask > 0] = 1
    mask = remove_too_small_pix(mask, pixels=30)
    l, c = ndimage.label(mask)
    l_masks = []
    for i in range(c):
        if np.sum(l == i + 1) < 160:
            m = np.zeros_like(mask)
            m[l == i + 1] = 1
            l_masks.append(m)
            mask[l == i + 1] = 0
    return mask, l_masks


def over_cars(
    imgb, mask, car_selected, rdraw=None, over_perc=0.8, neighbors=0, plot=False
):
    """Go over imgb and mask with car_selected and try to find regions with high overlap percentage.
    Those, with a shape close to a car are supposed cars.

    Args:
        imgb (numpy.ndarray): Gray scale (2D) image with background removed.
        mask (numpy.ndarray): Mask (2D) where cars are supposed to be.
        car_selected (numpy.ndarray): Mask (2D) of orientated car used to find cars in mask.
        rdraw (numpy.ndarray, optional): Rectangular (2D) mask around all supposed cars in mask.
        Used to control area of overlap. Defaults to None.
        over_perc (float, optional): Overlap percentage with car_selected and mask. Defaults to .8.
        neighbors (int, optional): Parameter to control to look around after a match to find a better
        one. Defaults to 0.
        plot (bool, optional): Whether or not to plot the different steps. Defaults to True.

    Returns:
        (list(numpy.ndarray)): List of the overlapped masks where are supposed cars.
    """

    igbb = imgb.copy()
    mtpb = mask.copy()
    pix_car = np.sum(car_selected)
    j = 0
    l_m = []
    # Go over the original mask and try to find overlaps with car_selected
    while j <= (igbb.shape[1] - car_selected.shape[1]):
        i = 0
        while i <= (igbb.shape[0] - car_selected.shape[0]):
            check_mask = True
            if rdraw is not None:
                # Check that the car_selected is in rdraw before checking overlap with mask
                roverlap = (
                    rdraw[
                        i : (car_selected.shape[0] + i), j : (car_selected.shape[1] + j)
                    ]
                    + car_selected
                )
                if np.sum(roverlap > 1) < 0.8 * pix_car:
                    check_mask = False
            if check_mask:
                overlap = mtpb.copy()
                overlap = (
                    mtpb[
                        i : (car_selected.shape[0] + i), j : (car_selected.shape[1] + j)
                    ]
                    + car_selected
                )
                # Check if detected
                if np.sum(overlap > 1) > over_perc * pix_car:
                    # Try to find a better match in the neighbourhood by moving to the right and to the bottom
                    # Removed the feature
                    max_find = (np.sum(overlap), i, j, overlap)
                    im, jm, om = max_find[1], max_find[2], max_find[3]
                    mtpb[
                        im : (car_selected.shape[0] + im),
                        jm : (car_selected.shape[1] + jm),
                    ][om > 1] = 0
                    m = np.zeros_like(mtpb)
                    m[
                        im : (car_selected.shape[0] + im),
                        jm : (car_selected.shape[1] + jm),
                    ][om > 1] = 1
                    l_m.append(m)
                    if plot:
                        plt.imshow(mtpb)
                        plt.show()
                    # Try to create openings in mask
                    mtpb = binary_opening(mtpb, iterations=2)
                    # Remove artefacts from the openings
                    mtpb = remove_too_small_pix(mtpb, 50)
                    if plot:
                        plt.imshow(mtpb)
                        plt.show()
                    # Aplly the result on igbb
                    igbb[mtpb == 0] = 0
            i += 1
        j += 1
    return l_m


def remove_f_boundaries(imgb, mask):
    """Apply felzenszwalb on imgb and remove the found boundaries. Pretty conservative.

    Args:
        imgb (numpy.ndarray): Image (2D) grayscale without background (filtered by mask).
        mask (numpy.ndarray): Mask of imgb.

    Returns:
        (numpy.ndarray, numpy.ndarray): imgb and mask with found boundaries removed.
    """
    segments = felzenszwalb(imgb, scale=200, sigma=0.5, min_size=100)
    bd = find_boundaries(segments)
    imgb[bd > 0] = 0
    mask[bd > 0] = 0
    return imgb, mask


def find_small_segment(cbox):
    """From a list of points drawing a rectangle, find the points drawing the smallest
    and highest side of it.

    Args:
        cbox (list(np.array)): List of arrays, each array being a points in space.

    Returns:
        (np.array, np.array): Tuple of the two points found.
    """
    f_dist = lambda a, b: np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    segments = [[cbox[i], cbox[i - 1]] for i in range(len(cbox))]
    distances = np.array([f_dist(a, b) for a, b in segments])
    inds = distances.argsort()
    s1 = segments[inds[0]]
    s2 = segments[inds[1]]
    higher = s1
    # higher, as working with matrix indices, means smaller y value
    if all([s1[0][1] < x[1] for x in s2]) or all([s1[1][1] < x[1] for x in s2]):
        higher = s1
    else:
        higher = s2
    return higher


def find_angle(pts, cbox):
    """From pts a list of points drawing the smallest and highest segment of a rectangle,
    and cbox the list of points drawing the rectangle, find the angle to use to oriente the car
    overlapping mask.

    Args:
        pts ((np.array, np.array)): Points of the smallest and highest segment in space.
        cbox (list(np.array)): Points drawing a rectangle.

    Returns:
        (float, bool): Angle to orient the car and if the segment was on the left of the
        rectangle or not.
    """
    # find angle formed by smallest segment
    highp = pts[0] if pts[0][1] < pts[1][1] else pts[1]
    lowp = pts[0] if pts[0][1] > pts[1][1] else pts[1]
    # Flat vector used to compute the angle
    vflat = np.array([highp[0] + 1, highp[1]]) - highp
    vangled = lowp - highp

    cosang = np.dot(vflat, vangled)
    sinang = np.linalg.norm(np.cross(vflat, vangled))
    angle = np.arctan2(sinang, cosang)
    angle = np.rad2deg(angle)
    # Check if on two points in p is the one at max left among points in cbox
    left = False
    for p in pts:
        if all([p[0] <= x[0] for x in cbox]):
            left = True
    # Applying correction to generate car mask from angle
    if angle < 90:
        angle = 180 - angle
    else:
        angle = 360 - angle
    return angle, left


def boundaries_to_remove(l_masks, plot=False):
    if len(l_masks) == 0:
        print("l_masks must contain at least a single mask to process.")
        remove = None
    else:
        lm = np.zeros_like(l_masks[0])
        for i, l in enumerate(l_masks):
            lm = lm + (l * (i + 1))
        boundaries = skimage.segmentation.find_boundaries(lm)
        if plot:
            plt.imshow(boundaries)
            plt.show()
        remove = np.zeros_like(l_masks[0])
        remove[boundaries > 0] = 1
        if plot:
            plt.imshow(remove)
            plt.show()
        return remove


def process_label(image, image_gray, mask, labels, label_id, plot=False):
    img = image.copy()
    imgg = image_gray.copy()
    pmask = mask.copy()

    l_masks = []
    # Crop image and mask to process label_id
    ii, obj_indices = ndimage.find_objects(labels)[label_id]
    ic = img[ii, obj_indices, :]
    ig = imgg[ii, obj_indices]
    mtp = pmask[ii, obj_indices]
    mtp = np.where(mtp > 0, 1, 0)
    igb = ig.copy()
    igb[mtp == 0] = 0
    if plot:
        plt.imshow(igb)
        plt.show()

    # Remove small part which are kept in cell, mostly from other labels
    mtp = remove_too_small_pix(mtp, 50)
    igb[mtp == 0] = 0
    if plot:
        plt.imshow(igb)
        plt.show()
    if np.sum(mtp) < 240:
        l_masks.append(mtp)
        mtpb = np.zeros_like(mtp)
    else:
        # Try to separate small part of the mask
        mtpb, l_m = open_remove(igb, iters=5)
        l_masks = l_masks + l_m
        if plot:
            plt.imshow(mtpb)
            plt.show()
        igb[mtpb == 0] = 0
        if plot:
            plt.imshow(igb)
            plt.show()

    # Check if there is enough packed pixels to contain close to two cars
    # Threshold set as one car is around 150 pix, then two with a small margin
    # should be higher than 240
    if np.sum(mtpb) < 240:
        # add remaining mask
        l_masks.append(mtpb)
    else:

        # Find the orientation of the cars
        angle, rbox, cbox = find_cars_orientation(ic, mtpb, False)
        # Generate the mask box onto which the cars should be
        rbox = np.sum(rbox, axis=-1)
        rbox[rbox > 0] = 1
        if plot:
            plt.imshow(rbox)
            plt.show()

        # Find the smallest lines with the higher point in array
        pts = find_small_segment(cbox)
        angle, left = find_angle(pts, cbox)

        car_selected = generate_car_mask(angle)
        if plot:
            plt.imshow(car_selected)
            plt.show()

        if left:
            l_m1 = over_cars(igb, mtpb, car_selected, rbox)
            l_masks = l_masks + l_m1
            if plot:
                for l in l_m1:
                    plt.imshow(l)
                    plt.show()

        else:
            igbf = np.flip(igb, axis=1)
            mtpbf = np.flip(mtpb, axis=1)
            carf = np.flip(car_selected, axis=1)
            rboxf = np.flip(rbox, axis=1)

            l_m1 = over_cars(igbf, mtpbf, carf, rboxf)
            # need to flip masks back
            l_m1 = [np.flip(x, axis=1) for x in l_m1]
            l_masks = l_masks + l_m1

            if plot:
                for l in l_m1:
                    plt.imshow(l)
                    plt.show()
    if plot:
        for l in l_masks:
            plt.imshow(l)
            plt.show()

    remove_mask = boundaries_to_remove(l_masks, plot)

    result = reduce(lambda a, b: a + b, l_masks)
    if plot:
        plt.imshow(result)
        plt.show()

    result[remove_mask > 0] = 0
    if plot:
        plt.imshow(result)
        plt.show()
    return result
