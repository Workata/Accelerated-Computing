"""
    This module demonstrates different filters. This implementation of those filters use CPU as a resource.
"""
# * imports
from PIL import Image
from datetime import datetime
import pandas as pd
from Utils import save_image, open_image, create_image, get_pixel

# * samples
DIR_IMG_OUTPUT = "ImgOutput"
SMALL_IMG = "tim-swaan-eOpewngf68w-unsplash_small.jpg"
MEDIUM_IMG = "tim-swaan-eOpewngf68w-unsplash_medium.jpg"
BIG_IMG = "tim-swaan-eOpewngf68w-unsplash_big.jpg"
BIGGEST_IMG = "tim-swaan-eOpewngf68w-unsplash_biggest.jpg"

EXCEL_NAME =  "OUTPUT_CPU.xlsx"

# * number of tests
N = 3

def grayscale_filter(image: Image):
    """
    This function filters an image using grayscale filter.

    About grayscale filter:
        Grayscale is a range of monochromatic shades from black to white.
        Therefore, a grayscale image contains only shades of gray and no color.

    Args:
        image (Image): Image to filter.

    Returns:
        Image: Image after grayscale filtering.
        float: total time of filtering.
    """
    width, height = image.size
    new_image = create_image(width, height)
    pixels = new_image.load()
    timestamp_1 = datetime.now()

    # * loop through each pixel in image
    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image, i, j)
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]

            # ! grayscale formula from wiki: 
            # ? https://en.wikipedia.org/wiki/Grayscale#Converting_colour_to_greyscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114) 

            pixels[i, j] = (int(gray), int(gray), int(gray))

    timestamp_2 = datetime.now()
    total_time = (timestamp_2 - timestamp_1).total_seconds()*1000
    return new_image, total_time

def color_filter(image : Image, color : str):
    """
    This function filters an image using color filter.

    About color filter:
        This filter allows through only choosen (red/green/blue) color.

    Args:
        image (Image): Image to filter.
        color (str): Color of filter (red, green, blue).

    Returns:
        Image: Image after choosen color filtering.
        float: total time of filtering.
    """
    width, height = image.size
    new_image = create_image(width, height)
    pixels = new_image.load()
    timestamp_1 = datetime.now()

    # * loop through each pixel in image
    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image, i, j)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            if(color == 'Red'):
                x = red
                y = 0
                z = 0

            if(color == 'Green'):
                x = 0
                y = green
                z = 0

            if(color == 'Blue'):
                x = 0
                y = 0
                z = blue

            pixels[i, j] = (int(x), int(y), int(z))

    timestamp_2 = datetime.now()
    total_time = (timestamp_2 - timestamp_1).total_seconds()*1000
    return new_image, total_time


def negative_filter(image: Image):
    """
    This function filters an image using negative filter.

    About negative filter:
        A positive image is a normal image. A negative image is a total inversion,
        in which light areas appear dark and vice versa. A negative color image is
        additionally color-reversed, with red areas appearing cyan, greens
        appearing magenta, and blues appearing yellow, and vice versa.

    Args:
        image (Image): Image to filter.

    Returns:
        Image: Image after negative filtering.
        float: total time of filtering.
    """
    width, height = image.size
    new_image = create_image(width, height)
    pixels = new_image.load()
    timestamp_1 = datetime.now()

    # * loop through each pixel in image
    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image, i, j)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            # * complement to each channel
            pixels[i, j] = (int(255-red), int(255-green), int(255-blue))

    timestamp_2 = datetime.now()
    total_time = (timestamp_2 - timestamp_1).total_seconds()*1000
    return new_image, total_time

def add_new_row_to_report(report : dict, test_img_name : str,
 filter_name : str, total_time : float):
    """
    This function adds given data to the dictionary (report).

    Args:
        report (dict): report that later will be converted to the excel.
        test_img_name (str): name of the image that is being tested.
        filter_name (str): name of the filter that is being tested.
        total_time (float): total time of filtering.

    Returns:
        dict: Report after update.
    """

    report['img_name'].append(test_img_name)
    report['filter_name'].append(filter_name)
    report['total_time'].append(total_time)
    return report

def test_filter(test_img_name : str, report : dict, filter_name : str, filter_option : str):
    """
    This function 

    Args:
        test_img_name (str): name of the iamge that is being tested.
        report (dict): report that later will be converted to the excel.
        filter_name (str): name of the filter that is being tested.
        filter_option (str): name that indicates the selected color filtering option.
        
    Returns:
        dict: Report after update.
    """
    
    test_img = open_image(test_img_name)

    for i in range(0, N):
        if filter_name == 'grayscale':
            filtered_img_grayscale, total_time = grayscale_filter(test_img)
            report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
            save_image(filtered_img_grayscale, f'./{DIR_IMG_OUTPUT}/grayscale_test_cpu.jpg')
        
        if filter_name == 'negative':
            filtered_img_negative, total_time = negative_filter(test_img)
            report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
            save_image(filtered_img_negative, f'./{DIR_IMG_OUTPUT}/negative_test_cpu.jpg')
            
        if filter_name == 'color':
            filtered_img_color, total_time = color_filter(test_img, filter_option)
            report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
            save_image(filtered_img_color, f'./{DIR_IMG_OUTPUT}/color_test_cpu.jpg')
    
    return report

if __name__ == "__main__":
    """
        SMALL_IMG - 640x427, 93.9 KB
        MEDIUM_IMG - 1920x1280, 838.8 KB
        BIG_IMG - 2400x1600, 1.24 MB
        BIGGEST_IMG - 5472x3648, 5.6 MB
    """

    report = {'img_name': [], 'filter_name': [], 'total_time': []}
    # * SMALL
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'color', filter_option = 'Red')
    # * MEDIUM
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'color', filter_option = 'Red')
    # * BIG
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'color', filter_option = 'Red')
    # * BIGGEST
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'color', filter_option = 'Red')

    data = pd.DataFrame(report)
    data.to_excel(EXCEL_NAME)
