
import numpy
import os
import sys

from PIL import Image

def main():

    folder_dir = sys.argv[1]
    pixel_dimensions = int(sys.argv[2])
    width = int(sys.argv[3])
    height = int(sys.argv[4])


    png_paths = []

    print('Looking for PNGS in folder %s' % folder_dir)

    for root, dirs, files in os.walk(folder_dir):
        for file in files:
            if file.endswith('.png'):
                png_paths.append(os.path.join(root, file))

    print('Found %s images...' % str(len(png_paths)))

    mosaic_width = pixel_dimensions * width
    mosaic_height = pixel_dimensions * height

    print('Creating an image with dimensions: %s x %s' % (mosaic_width, mosaic_height))

    mosaic = numpy.zeros((mosaic_height, mosaic_width))
    count = 0

    for j in range(height):
        j1 = (height - j) * pixel_dimensions
        j0 = (height - j - 1) * pixel_dimensions
        for i in range(width):
            i0 = i * pixel_dimensions
            i1 = (i + 1) * pixel_dimensions
            mosaic[j0:j1, i0:i1] = numpy.asarray(Image.open(png_paths[count]).convert('L'))
            count += 1

    im = Image.fromarray(mosaic).convert('RGB')
    im.save('output.png', optimize=True)

if __name__ == '__main__':
    main()