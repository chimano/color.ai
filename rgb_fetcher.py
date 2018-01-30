# from io import BytesIO, StringIO
# from urllib.request import urlopen
# from zipfile import ZipFile
import csv
from PIL import Image
import glob
from ReadingRGB import *

# from StringIO import StringIO

colors = {"black", "white", "grey", "brown", "green", "blue", "yellow", "orange", "red", "purple", "pink"}

base_url = "https://cvhci.anthropomatik.kit.edu/~bschauer/datasets/google-512/orig/"

image_color = []

for color in colors:
	tail = color + "+color"
	path = "pictures/" + tail + "/*jpeg"

	for filename in glob.glob(path):
		img = Image.open(filename)
		# pix = img.load()
		pix_val = list(img.getdata())
		# rgb_img = img.convert('RGB')
		r,g,b = AveragePixelRGB(pix_val)
		image_color.append([r,g,b,color])
		print(r,g,b, color)
		
	csv_name = color + ".csv"
	with open(csv_name, 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for individual_color in image_color:
			spamwriter.writerow(individual_color)