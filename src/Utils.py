from PIL import Image

DIR_IMG_INPUT = "ImgInput"

def open_image(img_name: str):
	"""
	Open an image using Image.open(...) method from PIL lib.

	Args:
		path (str): Path to the image that you want to open.

	Returns:
		Image: Loaded image.
	"""
	path = f"{DIR_IMG_INPUT}/{img_name}"
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