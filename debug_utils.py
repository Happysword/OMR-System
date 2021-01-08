import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
import io_utils

args = io_utils.get_command_line_args()
__DEBUG__ = args.debug


def debug_imshow(image, title=None):
    if __DEBUG__:
        if title is not None:
            cv2.imshow(f"DEBUG: {title}", image)
        else:
            cv2.imshow("DEBUG", image)
        cv2.waitKey()


def debug_print(*args, sep=' ', end='\n'):
    if __DEBUG__:
        print(*args, sep=sep, end=end, flush=True)


# Show the figures / plots inside the notebook
def debug_show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    if __DEBUG__:
        images_number = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1, images_number + 1)]
        figure = plt.figure()
        n = 1
        for image, title in zip(images, titles):
            a = figure.add_subplot(1, images_number, n)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        figure.set_size_inches(np.array(figure.get_size_inches()) * images_number)
        plt.show()


def debug_show_histogram(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    if __DEBUG__:
        plt.figure()
        img_histogram = histogram(img, nbins=256)

        bar(img_histogram[1].astype(np.uint8), img_histogram[0], width=0.8, align='center')
