from PIL import Image
import time

def open_image(path):
	image = Image.open(path)
	return image

def save_image(image, path):
	image.save(path, 'png')

def create_image(i, j):
	"""
	args
	"""
	image = Image.new("RGB", (i, j), "white")
	return image

def greyscale_filter(image):
	width, height = image.size
	new_image = create_image(width, height)
	pixels = new.load()
	t0=time.clock()

	for i in range(width):
		for j in range(height):
			pixel = get_pixel(image, i, j)
			red =   pixel[0]
			green = pixel[1]
			blue =  pixel[2]

			gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

			pixels[i, j] = (int(gray), int(gray), int(gray))

	t1=time.clock()
	print("Processing time to convert to B&W is",t1-t0,"s")
	return new_image