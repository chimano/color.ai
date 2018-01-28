from io import BytesIO, StringIO
from urllib.request import urlopen
from zipfile import ZipFile
import csv
from PIL import Image
import glob
from ReadingRGB import *
# from StringIO import StringIO

colors = {"black", "white", "grey", "brown", "green", "blue", "yellow", "orange", "red", "purple", "pink"}

base_url = "https://cvhci.anthropomatik.kit.edu/~bschauer/datasets/google-512/orig/"

# def get_average_color(x,y, n, image):
# 	r, g, b = 0, 0, 0
# 	count = 0
# 	for s in range(x, x+n+1):
# 		for t in range(y, y+n+1):
# 			pixlr, pixlg, pixlb = image[s, t]
# 			r += pixlr
# 			g += pixlg
# 			b += pixlb
# 			count += 1
# 	return ((r/count), (g/count), (b/count))

line = []

for color in colors:
	tail = color + "+color"
	path = "pictures/" + tail + "/*jpeg" 	

	for filename in glob.glob(path):
		img = Image.open(filename)
		pix = img.load()
		pix_val = list(img.getdata())
		rgb_img = img.convert('RGB')
		r, g, b = rgb_img.getpixel((0, 0))
		print (r,g,b)
		r,g,b = AveragePixelRGB(pix_val)
		line.append([r,g,b,color])

	csv_name= color + ".csv"
	with open(csv_name, 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for individual_color in line:
			spamwriter.writerow(individual_color)