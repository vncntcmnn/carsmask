import argparse
import numpy as np
from pathlib import Path
from skimage.io import imread
from cars.mask import CarsMask


def check_parameters(image_path, mask_path):
    ok = True
    if not Path(image_path).is_file():
        print("Please use a valid path to an image.")
        ok = False
    if not Path(mask_path).is_file():
        print("Please use a valid path to a mask.")
        ok = False
    return ok


def check_image_mask(image_path, mask_path):
    img = imread(image_path)
    pmask = imread(mask_path)
    ok = True
    if img.shape[0] != pmask.shape[0]:
        ok = False
        print("Image and mask do not have the same shape, those cannot be processed.")
    if img.shape[1] != pmask.shape[1]:
        ok = False
        print("Image and mask do not have the same shape, those cannot be processed.")
    if img.shape[2] != 3:
        ok = False
        print("Image does not have 3 channels as it should, it cannot be processed.")
    if len(pmask.shape) == 3:
        if pmask.shape[2] != 1:
            ok = False
            print(
                "Mask does not have a single channel as it should (black and white"
                " image, 255 value as white pixel), it cannot be processed."
            )
        else:
            # Need to reshape
            pmask = pmask[:, :, 0]
    return img, pmask, ok


def generate_save_path(mask_path):
    p = Path(mask_path)
    name = p.stem
    parent = p.parent
    name = name + "_processed.tif"
    return str(parent / name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script aims to post-process cars masks obtained from"
        " Unet segmentation of satellite imagery. A color image, 3 channels, and"
        " a mask, grayscale black and white, are needed to do so. Those two should"
        " have the same height and width to be processed."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file on which mask has been computed. Should be a 3"
        " channels image.",
    )
    parser.add_argument(
        "mask_path",
        type=str,
        help="Mask computed on image_path image. Should be a grayscale black"
        " (0 value) and white (255 value) image, with black used as background.",
    )
    parser.add_argument(
        "--save",
        help="Save the processed mask in the same folder as mask_path,"
        " with 'processed' added at the end of the name.",
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        help="Plot the different steps of the splitting process of the"
        " 'big' labels. There might be a lot of plots, and closing the current"
        " plot window is need to display the next one.",
        action="store_true",
    )

    args = parser.parse_args()
    check = check_parameters(args.image_path, args.mask_path)

    if check:
        img, pmask, ok = check_image_mask(args.image_path, args.mask_path)
        if ok:
            cmask = CarsMask(img, pmask)
            cmask.remove_small_objects(40)
            cmask.get_labels_to_split(shape_prod=None, pix=150)
            cmask.split_labels(args.plot)
            if args.save:
                save_path = generate_save_path(args.mask_path)
                cmask.save_processed_mask(save_path)
