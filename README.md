# Post-processing of cars masks obtained from segmentation of satellite imagery

This project aims to post-process segmentation masks obtained from satellite imagery and a Unet architecture. When multiple cars are close to each other, they are usually labelled as a single car when using scipy.ndimage. Thus it can be complicated to count the right number of cars in the image.
Multiple approaches can be applied to solve the issue, a few:
* Checking each "big" label, computing the number of pixels in the label's mask and divide it by the number of pixel on average for a single car.
* Using "classic" segmentation tools from open-cv or scikit-image, filters, etc. to split the cars and get new labels.
* Finding the orientation of the parked cars, and sliding a car mask over it to detect overlapping area.

The last one has been implemented, with far from perfect results, as the first is prone to errors and second one led to poor results.

## Usage

Project made with Python 3.8.5.

From the project directory:

1. Install the requirements:

    ```
    $ pip install -r requirements.txt
    ```
2. Run post-processing:

    ```
    $ python main.py --save image_path mask_path
    ```

    * **--save** (option): Save the results next to mask_path with same filename but `processed` added at the end.
    * **--plot** (option): Plot the different steps of the splitting process of the 'big' labels. there might be a lot of plots, and closing the current plot window is needed to display the next one.
    * **image_path** (str): Path to the image on which the mask has been computed. Has to be a 3 channels image.
    * **mask_path** (str): Path to the mask computed from image_path. Has to be a grayscale image with black (background, 0 value) and white (car class, 255 value).


    ### Examples:

    To process a single mask:
    ```
    $ python main.py --save /Users/random/segmentation/image.tif /Users/random/segmentation/mask.tif
    ```
