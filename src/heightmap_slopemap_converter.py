import numpy as np
import sys

from PIL import Image


def main():

    img_w, img_h = int(sys.argv[1]), int(sys.argv[2])
    min_h, max_h = int(sys.argv[3]), int(sys.argv[4])
    cell_size = float(sys.argv[5])
    height_diff = max_h - min_h

    height_map = Image.open(sys.argv[6]).convert('L')
    height_arr = []
    for x in range(img_w):
        height_row = []
        for y in range(img_h):
            height_row.append(min_h + height_map.getpixel((x, y)) / 255.0 * height_diff)

        height_arr.append(height_row)

    def calc_slope(x, y):
        max_slope = 0.0
        for i in range(-1, 2):
            xPos = x + i
            if 0 < xPos < img_w:
                for j in range(-1, 2):
                    yPos = y + j
                    if 0 < yPos < img_h:
                        slope_val = np.degrees(np.arctan(abs(height_arr[xPos][yPos] - height_arr[x][y]) / cell_size))
                        max_slope = max(max_slope, slope_val)
        return max_slope

    slope_map = np.zeros((img_h, img_w))

    for y in range(img_h):
        for x in range(img_w):
            slope_map[y][x] = calc_slope(x, y)

    slope_map = slope_map / 45.0  # Not Possible to farm on a slope that is greater than 45

    result = Image.fromarray((slope_map * 255.0).astype(np.uint8))
    print(result.size)
    result.save(sys.argv[7])


if __name__ == '__main__':
    main()
