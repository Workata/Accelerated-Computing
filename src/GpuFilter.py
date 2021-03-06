"""
    This module demonstrates different filters. This implementation of those filters use GPU as a main resource.
"""
# * imports
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy 
from datetime import datetime
from Utils import save_image, open_image
import pandas as pd

# * samples
DIR_IMG_OUTPUT = "ImgOutput"
SMALL_IMG = "tim-swaan-eOpewngf68w-unsplash_small.jpg"
MEDIUM_IMG = "tim-swaan-eOpewngf68w-unsplash_medium.jpg"
BIG_IMG = "tim-swaan-eOpewngf68w-unsplash_big.jpg"
BIGGEST_IMG = "tim-swaan-eOpewngf68w-unsplash_biggest.jpg"

EXCEL_NAME =  "OUTPUT_GPU.xlsx"

# * number of tests
N = 3

def grayscale_filter(image : Image):
    """
    This function filters an image using grayscale filter.

    About grayscale filter:
        Grayscale is a range of monochromatic shades from black to white.
        Therefore, a grayscale image contains only shades of gray and no color.

    Args:
        image (Image): Image to filter.

    Returns:
        Image: Image after grayscale filtering.
        float: time of memory allocation.
        float: time of operations in kernel.
        float: time of getting data back from gpu.
        float: total time of filtering.
    """

    total_timestamp_1 = datetime.now()
 
    image_pixels = numpy.array(image)
    image_pixels = image_pixels.astype(numpy.float32) # * Change data type
 
    alloc_timestamp_1 = datetime.now()

    device_image_pixels = cuda.mem_alloc(image_pixels.nbytes)  # * Allocate memory on the device, nbytes - number of bytes
    cuda.memcpy_htod(device_image_pixels, image_pixels)        # * Transfer the data to the GPU
 
    alloc_timestamp_2 = datetime.now()
 
    kernel_timestamp_1 = datetime.now()
 
    # * Kernel declaration - kernel grid and block size - https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
    # ! 1D-indexing
    BLOCK_SIZE = 1024
    block = (1024, 1, 1)    # * 1024 threads in block (block size: 1024x1x1)
    check_size = numpy.int32(image.size[0] * image.size[1]) # width * height of image
    grid = (int(image.size[0] * image.size[1] / BLOCK_SIZE) + 1, 1, 1) # * grid size (number_of_blocks, 1, 1)
 
    # * Kernel definition - write CUDA C code
    kernel = """
 
    __global__ void grayscaleFilter( float * inputImage, int check ){

        int subpixelIndex = (threadIdx.x ) + blockDim.x * blockIdx.x ; // subpixelIndex = index of subpixel (pointing at red SP) -> [... 'r, g, b','r, g, b' ...]
        if(subpixelIndex *3 < check*3)
        {
            int value = 0.299 * inputImage[subpixelIndex*3] + 0.587 * inputImage[subpixelIndex*3 + 1] + 0.114 * inputImage[subpixelIndex*3 + 2];
            inputImage[subpixelIndex*3] = value;      // red subpixel
            inputImage[subpixelIndex*3+1] = value;    // green subpixel
            inputImage[subpixelIndex*3+2] = value;    // blue subpixel
        }
    }
    """

    # * Compile and get kernel function
    mod = SourceModule(kernel)

    func = mod.get_function("grayscaleFilter")
    func(device_image_pixels, check_size, block = block, grid = grid)
 
    kernel_timestamp_2 = datetime.now()
 
    # * Get back data from gpu
    back_data_timestamp_1 = datetime.now()   
    grayscale_image_pixels = numpy.empty_like(image_pixels)         # * Return a new array with the same shape and type as a given array
    cuda.memcpy_dtoh(grayscale_image_pixels, device_image_pixels)   # * Fetch the data back from the GPU and display it
    grayscale_image_pixels = (numpy.uint8(grayscale_image_pixels))  # * Change data type
    back_data_timestamp_2 = datetime.now()
 
    grayscale_image = Image.fromarray(grayscale_image_pixels, mode ="RGB") # * Convert pixels to image
     
    total_timestamp_2 = datetime.now()
 
    # * Calculate total time operations
    alloc_time = (alloc_timestamp_2 - alloc_timestamp_1).total_seconds()*1000
    kernel_time = (kernel_timestamp_2 - kernel_timestamp_1).total_seconds()*1000
    back_data_time = (back_data_timestamp_2 - back_data_timestamp_1).total_seconds()*1000
    total_time = (total_timestamp_2 - total_timestamp_1).total_seconds()*1000

    return grayscale_image, alloc_time, kernel_time, back_data_time, total_time


def color_filter(image : Image, color_number: int):
    """
    This function filters an image using color filter.

    About color filter:
        This filter allows through only choosen (red/green/blue) color.

    Args:
        image (Image): Image to filter.
        color (int): Color of filter (0 - red, 1 - green, 2 - blue).

    Returns:
        Image: Image after choosen color filtering.
        float: time of memory allocation.
        float: time of operations in kernel.
        float: time of getting data back from gpu.
        float: total time of filtering.
    """

    total_timestamp_1 = datetime.now()
 
    image_pixels = numpy.array(image)
    image_pixels = image_pixels.astype(numpy.float32) # * Change data type
 
    alloc_timestamp_1 = datetime.now()
    
    device_image_pixels = cuda.mem_alloc(image_pixels.nbytes)  # * Allocate memory on the device
    cuda.memcpy_htod(device_image_pixels, image_pixels)        # * Transfer the data to the GPU
 
    alloc_timestamp_2 = datetime.now()
 
    kernel_timestamp_1 = datetime.now()
 
    # * Kernel declaration - kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024, 1, 1)
    check_size = numpy.int32(image.size[0] * image.size[1])
    grid = (int(image.size[0] * image.size[1] / BLOCK_SIZE) + 1, 1, 1)
 
    # * Kernel definition - write CUDA C code
    kernel = """
 
    __global__ void colorFilter( float * inputImage, int check, int color){
 
        int subpixelIndex = (threadIdx.x ) + blockDim.x * blockIdx.x; // subpixelIndex = index of subpixel (pointing at red SP) -> [... 'r, g, b','r, g, b' ...]
        if(subpixelIndex *3 < check*3)
        { 
            if(color == 0) // red
            {
                inputImage[subpixelIndex*3] = inputImage[subpixelIndex*3];
                inputImage[subpixelIndex*3+1] = 0;
                inputImage[subpixelIndex*3+2] = 0;
            }
            else if(color == 1) //green
            {
                inputImage[subpixelIndex*3] = 0;
                inputImage[subpixelIndex*3+1] = inputImage[subpixelIndex*3+1];
                inputImage[subpixelIndex*3+2] = 0;
            }
            else if(color == 2) //blue
            {
                inputImage[subpixelIndex*3] = 0;
                inputImage[subpixelIndex*3+1] = 0;
                inputImage[subpixelIndex*3+2] = inputImage[subpixelIndex*3+2];
            }
        }
    }
    """
    
    color = color_number
    color = numpy.int32(color)
    
    # * Compile and get kernel function
    mod = SourceModule(kernel)
    func = mod.get_function("colorFilter")
    func(device_image_pixels, check_size, block = block, grid = grid)
 
    kernel_timestamp_2 = datetime.now()
 
    back_data_timestamp_1 = datetime.now()
 
    # * Get back data from gpu
    color_image_pixels = numpy.empty_like(image_pixels)         # * Return a new array with the same shape and type as a given array
    cuda.memcpy_dtoh(color_image_pixels, device_image_pixels)   # * Fetch the data back from the GPU and display it
    color_image_pixels = (numpy.uint8(color_image_pixels))      # * Change data type
 
    back_data_timestamp_2 = datetime.now()
 
    color_image = Image.fromarray(color_image_pixels, mode ="RGB") # * Convert pixels to image
     
    total_timestamp_2 = datetime.now()
 
    # * Calculate total time operations
    alloc_time = (alloc_timestamp_2 - alloc_timestamp_1).total_seconds()*1000
    kernel_time = (kernel_timestamp_2 - kernel_timestamp_1).total_seconds()*1000
    back_data_time = (back_data_timestamp_2 - back_data_timestamp_1).total_seconds()*1000
    total_time = (total_timestamp_2 - total_timestamp_1).total_seconds()*1000

    return color_image, alloc_time, kernel_time, back_data_time, total_time
 

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
        float: time of memory allocation.
        float: time of operations in kernel.
        float: time of getting data back from gpu.
        float: total time of filtering.
    """

    total_timestamp_1 = datetime.now()
 
    image_pixels = numpy.array(image)
    image_pixels = image_pixels.astype(numpy.float32) # * Change data type
 
    alloc_timestamp_1 = datetime.now()

    device_image_pixels = cuda.mem_alloc(image_pixels.nbytes)  # * Allocate memory on the device
    cuda.memcpy_htod(device_image_pixels, image_pixels)        # * Transfer the data to the GPU
 
    alloc_timestamp_2 = datetime.now()
 
    kernel_timestamp_1 = datetime.now()
 
    # * Kernel declaration - kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024, 1, 1)
    check_size = numpy.int32(image.size[0] * image.size[1])
    grid = (int(image.size[0] * image.size[1] / BLOCK_SIZE) + 1, 1, 1)
 
    # * Kernel definition - write CUDA C code
    kernel = """
 
    __global__ void negativeFilter( float *inputImage, int check ){
 
        int subpixelIndex = (threadIdx.x ) + blockDim.x * blockIdx.x ; // subpixelIndex = index of subpixel (pointing at red SP) -> [... 'r, g, b','r, g, b' ...]
 
        if(subpixelIndex *3 < check*3)
        { 
            inputImage[subpixelIndex*3] = 255 - inputImage[subpixelIndex*3];
            inputImage[subpixelIndex*3+1] = 255 - inputImage[subpixelIndex*3+1];
            inputImage[subpixelIndex*3+2] = 255 - inputImage[subpixelIndex*3+2];
        }
    }
    """
 
    # * Compile and get kernel function
    mod = SourceModule(kernel)

    func = mod.get_function("negativeFilter")
    func(device_image_pixels, check_size, block = block, grid = grid)
 
    kernel_timestamp_2 = datetime.now()
 
    back_data_timestamp_1 = datetime.now()

    # * Get back data from gpu
    negative_image_pixels = numpy.empty_like(image_pixels)         # * Return a new array with the same shape and type as a given array
    cuda.memcpy_dtoh(negative_image_pixels, device_image_pixels)   # * Fetch the data back from the GPU and display it
    negative_image_pixels = (numpy.uint8(negative_image_pixels))   # * Change data type
 
    back_data_timestamp_2 = datetime.now()
 
    negative_image = Image.fromarray(negative_image_pixels, mode ="RGB")    # * Convert pixels to image
 
    total_timestamp_2 = datetime.now()
 
    # * Calculate total time of operations
    alloc_time = (alloc_timestamp_2 - alloc_timestamp_1).total_seconds()*1000
    kernel_time = (kernel_timestamp_2 - kernel_timestamp_1).total_seconds()*1000
    back_data_time = (back_data_timestamp_2 - back_data_timestamp_1).total_seconds()*1000
    total_time = (total_timestamp_2 - total_timestamp_1).total_seconds()*1000
 
    return negative_image, alloc_time, kernel_time, back_data_time, total_time


def add_new_row_to_report(report : dict, test_img_name : str, filter_name : str,
 alloc_time: float, kernel_time: float, back_data_time: float, total_time: float):
    """
    This function adds given data to the dictionary (report).

    Args:
        report (dict): report that later will be converted to the excel.
        test_img_name (str): name of the image that is being tested.
        filter_name (str): name of the filter that is being tested.
        alloc_time (float): time of memory allocation.
        kernel_time (float): time of operations in kernel.
        back_data_time (float): time of getting data back from gpu.
        total_time (float): total time of filtering.

	Returns:
		dict: Report after update.
    """

    report['img_name'].append(test_img_name)
    report['filter_name'].append(filter_name)
    report['alloc_time'].append(alloc_time)
    report['kernel_time'].append(kernel_time)
    report['back_data_time'].append(back_data_time)
    report['total_time'].append(total_time)
    return report

def test_filter(test_img_name : str, report : dict, filter_name : str, filter_option : int):
    """
	This function 

	Args:
		test_img_name (str): name of the iamge that is being tested.
        report (dict): report that later will be converted to the excel.
        filter_name (str): name of the filter that is being tested.
        filter_option (int): number that indicates the selected color filtering option.
        
	Returns:
		dict: Report after update.
	"""
    
    test_img = open_image(test_img_name)

    for i in range(0, N):
        if filter_name == 'grayscale':
            filtered_img_grayscale, alloc_time, kernel_time, back_data_time, total_time = grayscale_filter(test_img)
            report = add_new_row_to_report(report, test_img_name, filter_name, alloc_time, kernel_time, back_data_time, total_time)
            save_image(filtered_img_grayscale, f'./{DIR_IMG_OUTPUT}/grayscale_test_gpu.jpg')
        
        if filter_name == 'negative':
            filtered_img_negative, alloc_time, kernel_time, back_data_time, total_time = negative_filter(test_img)
            report = add_new_row_to_report(report, test_img_name, filter_name, alloc_time, kernel_time, back_data_time, total_time)
            save_image(filtered_img_negative, f'./{DIR_IMG_OUTPUT}/negative_test_gpu.jpg')
            
        if filter_name == 'color':
            filtered_img_color, alloc_time, kernel_time, back_data_time, total_time = color_filter(test_img, filter_option)
            report = add_new_row_to_report(report, test_img_name, filter_name, alloc_time, kernel_time, back_data_time, total_time)
            save_image(filtered_img_color, f'./{DIR_IMG_OUTPUT}/color_test_gpu.jpg')
    
    return report


if __name__ == "__main__":
    """
        SMALL_IMG - 640x427, 93.9 KB
        MEDIUM_IMG - 1920x1280, 838.8 KB
        BIG_IMG - 2400x1600, 1.24 MB
        BIGGEST_IMG - 5472x3648, 5.6 MB
    """

    report = {'img_name': [], 'filter_name': [], 'alloc_time': [], 'kernel_time': [], 'back_data_time': [], 'total_time': []}

    # * SMALL
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = SMALL_IMG, report = report, filter_name = 'color', filter_option = 0)
    # * MEDIUM
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = MEDIUM_IMG, report = report, filter_name = 'color', filter_option = 0)
    # * BIG
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = BIG_IMG, report = report, filter_name = 'color', filter_option = 0)
    # * BIGGEST
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'grayscale', filter_option = None)
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'negative', filter_option = None)
    report = test_filter(test_img_name = BIGGEST_IMG, report = report, filter_name = 'color', filter_option = 0)

    data = pd.DataFrame(report)
    data.to_excel(EXCEL_NAME)
