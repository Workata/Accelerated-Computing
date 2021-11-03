"""
    TODO docstring
"""
# * imports
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy 
from datetime import datetime

# * samples
DIR_IMG_INPUT = "ImgInput"
DIR_IMG_OUTPUT = "ImgOutput"
SMALL_IMG = "tim-swaan-eOpewngf68w-unsplash_small.jpg"
MEDIUM_IMG = "tim-swaan-eOpewngf68w-unsplash_medium.jpg"
BIG_IMG = "tim-swaan-eOpewngf68w-unsplash_big.jpg"
BIGGEST_IMG = "tim-swaan-eOpewngf68w-unsplash_biggest.jpg"

EXCEL_NAME =  "Output.xlsx"

# * number of tests
N = 3

def grayscale_filter(input_path, output_path):

    total_timestamp_1 = datetime.now()
 
    image = Image.open(input_path)
    image_pixels = numpy.array(image)
    image_pixels = image_pixels.astype(numpy.float32) # * change data type
 
    # getAndConvertT1 = time.clock()
 
    alloc_timestamp_1 = datetime.now()

    device_image_pixels = cuda.mem_alloc(image_pixels.nbytes)  # * allocate memory on the device
    cuda.memcpy_htod(device_image_pixels, image_pixels)        # * transfer the data to the GPU # * transfer the data to the GPU
 
    alloc_timestamp_2 = datetime.now()
 
    kernel_timestamp_1 = datetime.now()
 
    # * Kernel declaration - kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024, 1, 1)
    check_size = numpy.int32(image.size[0] * image.size[1])
    grid = (int(image.size[0] * image.size[1] / BLOCK_SIZE) + 1, 1, 1)
 
    # * Kernel definition - write CUDA C code
    kernel = """
 
    __global__ void grayscaleFilter( float * inputImage, int check ){

        int subpixelIndex = (threadIdx.x ) + blockDim.x * blockIdx.x ; // subpixelIndex = index of subpixel (pointing at red SP) -> [... 'r, g, b','r, g, b' ...]
        if(subpixelIndex *3 < check*3)
        {
       		int value = 0.21 * inputImage[subpixelIndex*3] + 0.71 * inputImage[subpixelIndex*3 + 1] + 0.07 * inputImage[subpixelIndex*3 + 2];
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
 
   
    back_data_timestamp_1 = datetime.now()

    # * Get back data from gpu
    grayscale_image_pixels = numpy.empty_like(image_pixels)         # * Return a new array with the same shape and type as a given array
    cuda.memcpy_dtoh(grayscale_image_pixels, device_image_pixels)   # * Fetch the data back from the GPU and display it
    grayscale_image_pixels = (numpy.uint8(grayscale_image_pixels))  # * Change data type
 
    back_data_timestamp_2 = datetime.now()
 
    # * Save image
    store_image_timestamp_1 = datetime.now()
    pil_im = Image.fromarray(grayscale_image_pixels, mode ="RGB")
 
    pil_im.save(output_path)
     
    total_timestamp_2 = datetime.now()
 
    # * calculate total time operations
    # getAndConvertTime = getAndConvertT1 - totalT0
    alloc_time = (alloc_timestamp_2 - alloc_timestamp_1).total_seconds()*1000
    kernel_time = (kernel_timestamp_2 - kernel_timestamp_1).total_seconds()*1000
    back_data_time = (back_data_timestamp_2 - back_data_timestamp_1).total_seconds()*1000
    store_image_time = (total_timestamp_2 - store_image_timestamp_1).total_seconds()*1000
    total_time = (total_timestamp_2 - total_timestamp_1).total_seconds()*1000
 
    print("Black and white image")
    print("Image size: ", image.size)
    # print("Time taken to get and convert image data: " ,getAndConvertTime)
    print("Time taken to allocate memory on the GPU: " , alloc_time)
    print("Kernel execution time: " , kernel_time)
    print("Time taken to get image data from GPU and convert it: " , back_data_time)
    print("Time taken to save the image: " , store_image_time)
    print("Total execution time : " , total_time)


if __name__ == "__main__":

    grayscale_filter(f"./{DIR_IMG_INPUT}/{SMALL_IMG}" , f"./{DIR_IMG_OUTPUT}/grayscale_test.jpg")

