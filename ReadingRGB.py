from PIL import Image

def AveragePixelRGB(pix_list):
    red = 0
    green = 0
    blue = 0
    for val in range(0,len(pix_list),8):
        red += pix_list[val][0]
        green += pix_list[val][1]
        blue += pix_list[val][2]
    total = len(pix_list)/8
    red /= total
    green /= total
    blue /= total
    red = int(round(red, 0))
    green = int(round(green, 0))
    blue = int(round(blue, 0))
    return [red, green, blue]
    
im = Image.open("photo.png",'r') #Can be many different formats.
pix = im.load()
pix_val = list(im.getdata())
rgb_im = im.convert('RGB')
r, g, b = rgb_im.getpixel((0, 0))

print (r,g,b)
print (AveragePixelRGB(pix_val))

