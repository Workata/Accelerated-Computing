"""
This module demonstrates different filters. This implementation of those filters use CPU as a resource.

TODOs:
	- Docstrings
"""
# * imports
from PIL import Image
from datetime import datetime
import pandas as pd

# * samples
DIR_IMG = "ImgInput"
SMALL_IMG = "tim-swaan-eOpewngf68w-unsplash_small.jpg"
MEDIUM_IMG = "tim-swaan-eOpewngf68w-unsplash_medium.jpg"
BIG_IMG = "tim-swaan-eOpewngf68w-unsplash_big.jpg"
BIGGEST_IMG = "tim-swaan-eOpewngf68w-unsplash_biggest.jpg"

# * number of tests
N = 3

def open_image(img_name: str):
	"""
	Open an image using Image.open(...) method from PIL lib.

	Args:
		path (str): Path to the image that you want to open.

	Returns:
		Image: Loaded image.
	"""
	path = f"{DIR_IMG}/{img_name}"
	image = Image.open(path)
	return image

def save_image(image: Image, path: str):
	"""
	Open an image using Image.save(...) method from PIL lib.

	Args:
		image (Image): Image to get pixel from.
		path (str): Path to the save location.

	Returns:
		None
	"""
	image.save(path)

def create_image(width: int, height: int):
	"""
	Create new image using Image.new(...) method from PIL lib.

	https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.new

	Args:
		width (int): width of an image
		height (int): height of an image

	Returns:
		Image: Newly created image.
	"""
	image = Image.new("RGB", (width, height), "white")
	return image

def get_pixel(image : Image, i : int, j : int):
	"""
	Get pixel of an image.

	Args:
		image (Image): Image to get pixel from.
		i (int): x-axis coordinate of pixel
		j (int): y-axis coordinate of pixel

	Returns:
		Tuple: Pixel of an image.
	"""
	width, height = image.size
	if i > width or j > height:
		return None
	pixel = image.getpixel((i, j))
	return pixel

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
	print(f"Grayscale filter time: {str(total_time)} ms")
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
				y = green - 255
				z = blue - 255

			if(color == 'Green'):
				x = red - 255
				y = green
				z = blue - 255

			if(color == 'Blue'):
				x = red - 255
				y = green - 255
				z = blue

			# * trim each channel <0, 255>
			x = max(x, 0)
			x = min(255, x)
			y = max(y, 0)
			y = min(255, y)
			z = max(z, 0)
			z = min(255, z)

			pixels[i, j] = (int(x), int(y), int(z))

	timestamp_2 = datetime.now()
	total_time = (timestamp_2 - timestamp_1).total_seconds()*1000
	print(f"Color filter ({color}) time: {str(total_time)} ms")
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
	print(f"Negative filter time: {str(total_time)} ms")
	return new_image, total_time

def add_new_row_to_report(report, test_img_name, filter_name, total_time):
	"""
	TODO Docstring
	"""
	report['img_name'].append(test_img_name)
	report['filter_name'].append(filter_name)
	report['total_time'].append(total_time)
	return report

def test_filter(test_img_name, report, filter_name, filter_option):
	"""
	TODO Docstring
	"""
	test_img = open_image(test_img_name)

	for i in range(0, N):
		if filter_name == 'grayscale':
			filtered_img_grayscale, total_time = grayscale_filter(test_img)
			report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
			save_image(filtered_img_grayscale, './ImgOutput/grayscale_test.jpg')
		
		if filter_name == 'negative':
			filtered_img_negative, total_time = negative_filter(test_img)
			report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
			save_image(filtered_img_negative, './ImgOutput/negative_test.jpg')
			
		if filter_name == 'color':
			filtered_img_color, total_time = color_filter(test_img, filter_option)
			report = add_new_row_to_report(report, test_img_name, filter_name, total_time)
			save_image(filtered_img_color, './ImgOutput/color_test.jpg')
	
	return report

if __name__ == "__main__":
	"""
		TODO docstring
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
	data.to_excel("Output.xlsx")

