# Chroma key
Computer Vision Algorithms and Systems
Objectives
1. Getting familiar with OpenCV 4.6.0
2. Reading, processing and displaying images using OpenCV 4.6.0
3. Understanding color spaces

Task One

Many color spaces are available to represent pixel values of a color image. Some are perceptually uniform and others are not. This task is to read an image and display the original color image and its components of a specified color space, such as CIE-XYZ, CIE-Lab, YCrCb and HSB. The color components will be displayed in gray-scale. The original image and its three components will be display in a single viewing window.

Task Two

A Chroma key is a technique used in film, television studio and photography to replace a portion of an image with a new image or to place a person, such as a newsreader, on a scenic background. The program will display in a single viewing window the photo of a person in front of a green screen, person extracted from the green screen photo with white background, scenic photo, photo with the person being in the scenic in a single viewing window.

Implementation
The program is named as “Chromakey” and will take
  
  a. one filename and one of the options –XYZ, -Lab,-YCrCb or -HSB as the command argument for Task One, e.g. Chromakey –XYZ|-Lab|-YCrCb|-HSB imagefile # task one or
  
  b. two filename as the command arguments for Task Two, e.g. Chromakey scenicImageFile greenScreenImagefile # task two
