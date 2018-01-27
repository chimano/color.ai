from PIL import Image
im = Image.open("dead_parrot.jpg") #Can be many different formats.
pix = im.load()
print (im.size)
rgb_im = im.convert('RGB')
r, g, b = rgb_im.getpixel((1, 1))
print (r,g,b)
