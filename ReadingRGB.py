from PIL import Image
import math

def AveragePixelRGB(pix_list):
    try:
        red = int(pix_list[0][0])
        green = int(pix_list[0][1])
        blue = int(pix_list[0][2])
    except:
        return 180, 180, 180
        
    for val in range(0,len(pix_list),10):
        red = get_two_average(red,pix_list[val][0])
        green = get_two_average(green,pix_list[val][1])
        blue = get_two_average(blue, pix_list[val][2])
    
    red = int(round(red))
    green = int(round(green))
    blue = int(round(blue))
    return red, green, blue
    
def get_two_average(color1, color2):
    square1 = color1*color1
    square2 = color2*color2
    partial_sum = square1 + square2
    partial_sum /= 2
    return  math.sqrt(partial_sum)