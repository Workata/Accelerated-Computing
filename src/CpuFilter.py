from PIL import Image
from datetime import datetime

def open_image(path):
	image = Image.open(path)
	return image

def save_image(image, path):
	image.save(path)

def create_image(i, j):
	"""
	args
	"""
	image = Image.new("RGB", (i, j), "white")
	return image

def get_pixel(image, i, j):
	width, height = image.size
	if i > width or j > height:
		return None
	pixel = image.getpixel((i, j))
	return pixel

def greyscale_filter(image : Image):
	"""
	greyscale
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
	print(f"Greyscale filter time: {str((timestamp_2 - timestamp_1).total_seconds()*1000)} ms")
	return new_image

if __name__ == "__main__":
	test_img = open_image('./ImgInput/joel-filipe-QwoNAhbmLLo-unsplash.jpg')
	filtered_img = greyscale_filter(test_img)
	save_image(filtered_img, './ImgOutput/test1.jpg')
