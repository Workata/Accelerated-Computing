"""
This module demonstrates different filters. This implementation of those filters use CPU as a resource.

TODOs:
    TODO dont comment code in main (add if, else (and static var) or input from console/file)
"""
from PIL import Image
from datetime import datetime

def open_image(path: str):
	"""
    Open an image using Image.open(...) method from PIL lib.

    Args:
        path (str): Path to the image that you want to open.

    Returns:
        Image: Loaded image.
	"""
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

	for i in range(width):
		for j in range(height):
			pixel = get_pixel(image, i, j)
			red = pixel[0]
			green = pixel[1]
			blue = pixel[2]

			gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

			pixels[i, j] = (int(gray), int(gray), int(gray))

	timestamp_2 = datetime.now()
	print(f"Grayscale filter time: {str((timestamp_2 - timestamp_1).total_seconds()*1000)} ms")
	return new_image

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

    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image, i, j)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            if(color == 'Red'):
                x = red
                y = green-255
                z = blue-255

            if(color == 'Green'):
                x = red-255
                y = green
                z = blue-255

            if(color == 'Blue'):
                x = red-255
                y = green-255
                z = blue

            x = max(x, 0)
            x = min(255,x)

            y = max(y, 0)
            y = min(255,y)

            z = max(z, 0)
            z = min(255,z)

            pixels[i, j] = (int(x), int(y), int(z))

    timestamp_2 = datetime.now()
    print(f"Color filter ({color}) time: {str((timestamp_2 - timestamp_1).total_seconds()*1000)} ms")
    return new_image


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

	for i in range(width):
		for j in range(height):
			pixel = get_pixel(image, i, j)
			red =   pixel[0]
			green = pixel[1]
			blue =  pixel[2]
			
			pixels[i, j] = (int(255-red), int(255-green), int(255-blue))

	timestamp_2 = datetime.now()
	print(f"Negative filter time: {str((timestamp_2 - timestamp_1).total_seconds()*1000)} ms")
	return new_image

if __name__ == "__main__":
	test_img = open_image('./ImgInput/joel-filipe-QwoNAhbmLLo-unsplash.jpg')

	filtered_img_grayscale = grayscale_filter(test_img)
	save_image(filtered_img_grayscale, './ImgOutput/grayscale_test.jpg')

	# filtered_img_color = color_filter(test_img, "Red")
	# save_image(filtered_img_color, './ImgOutput/color_test.jpg')
	
	# filtered_img_negative = negative_filter(test_img)
	# save_image(filtered_img_negative, './ImgOutput/negative_test.jpg')
