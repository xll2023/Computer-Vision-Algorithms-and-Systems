# CSCI935/CSCI435 (S223) Computer Vision Algorithms and Systems
# Assignment 1



import cv2
import numpy as np
import sys

def color_space_convert(color_space,image):
    """
    Function to convert image to specific color space
    and convert back to gray scale image
    """
    # user input option and split image from BGR to specific color space channel
    # consider HSB is euqal to HSV, so the input HSB or HSV is acceptable
    color_space_options = {
        '-XYZ': cv2.COLOR_BGR2XYZ,
        '-LAB': cv2.COLOR_BGR2LAB,
        '-YCRCB': cv2.COLOR_BGR2YCrCb,
        '-HSB': cv2.COLOR_BGR2HSV,
        '-HSV': cv2.COLOR_BGR2HSV
    }
    if color_space in color_space_options:
        conversion_code = color_space_options[color_space]
    color_space_image = cv2.cvtColor(image,conversion_code)
    c1, c2, c3 = cv2.split(color_space_image)

    # Normalize the special color channel to match RGB value ranges
    # RGB image pixel values are between 0 to 255
    # XYZ and YCrCb can cover the range as RGB, no normallization is needed

    # Normalize Lab: 
    # from 0<=L<=255, 1<=a<=255, 1<=b<=255 
    # to 0<=L<=255, 0<=a<=255, 0<=b<=255
    if conversion_code == 'cv2.COLOR_BGR2LAB':
        c2 = ((c2-1)*255/254).astype(np.uint8)
        c3 = ((c3-1)*255/254).astype(np.uint8)
    # Normalize HSB or HSV:
    # from 0<=H<=179  0<=S<=255, 0<=B or V<=255 
    # to 0<=H<=255  0<=S<=255, 0<=B or V<=255
    elif conversion_code == 'cv2.COLOR_BGR2HSV':
        c1 = (c1*255/179).astype(np.uint8)

    # merge each channel back to a 3D image in gray scale
    c1_3D = cv2.merge((c1,c1,c1))
    c2_3D = cv2.merge((c2,c2,c2))
    c3_3D = cv2.merge((c3,c3,c3))

    return c1_3D,c2_3D,c3_3D

def resize_image(image,desired_height,desired_width):
    """
    Function to resize image to desired dimension
    and keep the original aspect ratios unchanged
    """
    # get image height and width to caculate the original aspect ratios
    image_height,image_width = image.shape[:2]
    aspect_ratio_image = image_width / image_height

    # caculate the odesired aspect ratios and make a comparasion with original one
    # Determine the benchmark of the resized image based on the comparasion
    aspect_ratio_desired = desired_width / desired_height
    if aspect_ratio_image >  aspect_ratio_desired:  
        new_width = desired_width 
        new_height = int(new_width / aspect_ratio_image)
    else:
        new_height = desired_height
        new_width = int(new_height * aspect_ratio_image)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def remove_greenscreen(image):
    """
    Function to remove image green screen background
    """
    # Use HSB to extract colore space and use the bound range to create a mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_green = np.array([50, 43, 82])
    u_green = np.array([169, 255, 255])
    mask = cv2.inRange(hsv, l_green, u_green)

    # extract the green screen background with the mask and remove it from the image
    green_background=cv2.bitwise_and(image,image,mask=mask)
    image_nogreenscreen=image-green_background

    return image_nogreenscreen

def crop_image(contours_image,cropping_image):
    """
    Function to crop image based on the content contours
    """
    # use HSB color conversation to create the mask for contours detection 
    hsv = cv2.cvtColor(contours_image, cv2.COLOR_BGR2HSV)
    l_green = np.array([50, 43, 82])
    u_green = np.array([169, 255, 255])
    mask= cv2.inRange(hsv, l_green, u_green)
    mask =cv2.bitwise_not(mask)  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # make sure get the main object in the image with the contours
    c=max(contours,key=cv2.contourArea)

    # get boundary information and use them as coordinators to crop the image
    x, y, w, h = cv2.boundingRect(c)
    cropped_img=cropping_image[y:y+h,x:x+w]

    return cropped_img

def padding_image(image,size_src_image,padding_percentage_left,padding_percentage_top):
    """
    Function to add padding for image to match the dimension 
    """
    # Specify the dimensions of the desired size
    desired_width = size_src_image.shape[1]
    desired_height = size_src_image.shape[0]

    # Create a canvas using NumPy in desired size
    padded_image = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

    # Calculate the positions to place the original image within the padded image
    # padding_percentage to the left or top give options to determine the padding space percentage to the edge
    x_offset = int((desired_width - image.shape[1]) * padding_percentage_left)
    y_offset = int((desired_height - image.shape[0]) * padding_percentage_top)

    # Place the original image within the padded canvas
    padded_image[y_offset:y_offset+image.shape[0], 
                x_offset:x_offset+image.shape[1], :] = image
    
    return padded_image
    
def combine_image(t_l,t_r,b_l,b_r):
    """
    Function to combine 4 images in a single view window
    and make sure the combined image is not exceed the maximum size 1280 * 720 
    """
    # combine 2 images in the top row and 2 images in the bottom row
    top_row = np.hstack((t_l,t_r))
    bottom_row = np.hstack((b_l,b_r))
    # combine 2 rows vertically together to form the combined image
    combined_image = np.vstack((top_row, bottom_row))
    
    #get size information of combined image and set up the maximum dimension 
    height, width = combined_image.shape[:2]
    max_width = 1280
    max_height = 720
    # Determine whether the image exceeds the limited size
    # If the size exceed the limitation, then resize it, else return it
    if width > max_width or height > max_height:
        combined_image=resize_image(combined_image, max_height, max_width)

    return combined_image

def task1 (color_space,image_path):
    """
    Main function to execute task1 which display the original color image 
    and its components of a specified color space in gray scale in a single viewing window
    and make sure the combined image is not exceed the maximum size 1280 * 720 
    """
    # load the image
    image = cv2.imread(image_path)

    # get desired color space channels in gray scale separatly as c1,c2,c3
    # c1 (e.g. X, L, Y or H)
    # c2 (e.g. Y, a, Cr or S)
    # c3 (e.g. Z, b, Cb, or B/V)    
    c1,c2,c3=color_space_convert(color_space,image)
    # combine original image with components of a specified color space in one image and limit the size
    # check if the color space is HSB or HSV as the task1 request:
    # output HSB/HSV in order of [original image, B, H, S]
    if color_space == '-HSB' or color_space == '-HSV':
        combined_image = combine_image(image, c3, c1, c2)
    # for other color spaces, the output order is:
    # CIE-XYZ: [original image, X, Y, Z]
    # CIE-Lab: [original image, L, a, b]
    # CIE-YCrCb: [original image, Y, Cr, Cb]
    else:
        combined_image = combine_image(image, c1, c2, c3)

    # display the result 
    show_image(combined_image)

def task2(scenic_image_path,greenscreen_image_path):
    """
    Main function to execute task2 which is to extract the person in a green screen photo 
    and place the person in a scenic photo according to the following:
        a) The combined photo should be of the same size as the scenic photo, and
        b) The person should be aligned horizontally to the middle of the scenic photo 
    and display in a single viewing window the photo of 
        a person in front of a green screen, 
        person extracted from the green screen photo with white background, 
        scenic photo, 
        photo with the person being in the scenic
    and make sure the combined image is not exceed the maximum size 1280 * 720 
    """
    # load the scenic image file and green screen image file
    scenic=cv2.imread(scenic_image_path)
    image_gs = cv2.imread(greenscreen_image_path)

    # resize the green screen image based on the scenic image dimension and keep the aspect ratios
    image_gs=resize_image(image_gs,scenic.shape[0],scenic.shape[1])

    # make some padding space if the resized image can not match the scenic image dimension
    # the parameter (1,1) stands for padding 100% of the space to the top and left of the image, 
    # which locate the image to the botoom-right corner
    # This is to make sure the original image can be combined with other images without any gaps 
    image_padded=padding_image(image_gs,scenic,1,1)
    
    # remove the green screen out from the original image 
    image_gsremoved=remove_greenscreen(image_gs)
    
    # padding the background removed image to scenic dimension
    image_gsremoved_padded=padding_image(image_gsremoved,scenic,0.5,1)

    # use Numpy to fill out the background of the image with white color, which is value of 255
    image_whitebg=np.where(image_gsremoved_padded==0,255,image_gsremoved_padded)

    # crop the person boundary from the image without background in a rectangle shape
    cropped_img=crop_image(image_gs,image_gsremoved)

    # make the canvas to do some padding for the cropped image to match the dimension of the scenic image
    # make sure the cropped people image is located in the mildde of the padded canvas horizontally
    image_gsremoved_padded=padding_image(cropped_img,scenic,0.5,1)

    # use Numpy to fill out background of the padded image that people in the middle of the canvas horizontally with scenic image
    # as the canvas is the same dimension with scenic image and people image location in the combined image with scenic will be in the middle as well
    image_scenicbg=np.where(image_gsremoved_padded==0,scenic,image_gsremoved_padded)

    # combine the original image that properly padded, people image with white background, scenic image and the people image with scenic background in one image
    combined_image = combine_image(image_padded, image_whitebg, scenic, image_scenicbg)
    # resize the combined image to the scenic image dimension as request
    combined_image_resized = cv2.resize(combined_image, (scenic.shape[1], scenic.shape[0]))

    # display the result 
    show_image(combined_image_resized)

def show_image(image):
    """
    Function to display image
    :param image:
    :return:
    """
    # name the display window 
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # show the image in the window
    cv2.imshow('image', image)
    # infinity loop with 0 miliseconds wait after loop, press any key can stop the display
    cv2.waitKey(0)
    # destroy or close all windows at any time after exiting the script
    cv2.destroyAllWindows()

def parse_and_run():
    """
    Function to determine if the user input the designed arguments 
    and which task function will be executed
    """
    # check if the command agruments length is 3 including the '.py' file name
    # check if argument[2] is a file name with extension
    if len(sys.argv)== 3 and '.' in sys.argv[2]: 
        # check if the argument[1] is one of ["-XYZ", "-LAB", "-YCRCB", "-HSB", "-HSV"] to decide task 1 or task 2
        # use .upper() to make it input letters case unsensitive
        # consider HSB is euqal to HSV, so the input HSB or HSV is acceptable
        if sys.argv[1].upper() in ["-XYZ", "-LAB", "-YCRCB", "-HSB", "-HSV"]:           
            color_space= sys.argv[1]
            image= sys.argv[2]
            task1(color_space.upper(),image)
        # for argument[1] is a file name with extension, try to execute task 2
        elif '.' in sys.argv[1]:
            scenic_image= sys.argv[1]
            green_screen_image = sys.argv[2]
            task2(scenic_image,green_screen_image)
        # for else, give the prompt as following for user input
        else:
            print()
            print("If you are attempt to execute task 1.")
            print("Please only input one of [-XYZ, -LAB, -YCRCB, -HSB] for argument[1]!")
            print()
            print("If you are attempt to execute task 2.")
            print("Please input scenicImageFile(with extension) greenScreenImagefile(with extension)!")
            print()

    #if the arguments length is not equal to 3, then give the prompt to ask for correct input
    else:
        print()
        print("Attention! Please enter the command with only 3 arguments in following format (including space):")
        print()
        print("Task1:")
        print("Chromakey.py -XYZ|-Lab|-YCrCb|-HSB(choose one color space option including '-' ahead) imageFile(with extension)")
        print()
        print("Task2:")
        print("Chromakey.py scenicImageFile(with extension) greenScreenImagefile(with extension)")
        print()

# ensure that the parse_and_run() function is executed only when the script is run directly
# not when it's imported as a module into another script	
if __name__ == '__main__':
	parse_and_run()
