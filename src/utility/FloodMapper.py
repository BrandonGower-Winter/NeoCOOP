import numpy
import os
import sys
import math

from PIL import Image
from ECAgent.Environments import discreteGridPosToID

def main():

    max_h = float(sys.argv[3])
    heightmap = numpy.asarray(Image.open(sys.argv[1]).convert('L')) / 255.0 * max_h
    isWaterArr = numpy.asarray(Image.open(sys.argv[2]).convert('L')) > 0.005

    output = numpy.zeros(heightmap.shape)

    flood_cell_divide = int(sys.argv[4])
    flood_divide_ratio = len(output[0]) // flood_cell_divide
    print(flood_divide_ratio)
    avg_water_heights = []

    for y_block in range(math.ceil(len(output) / flood_cell_divide)):
        for x_block in range(math.ceil(len(output[y_block]) / flood_cell_divide)):

            total = 0.0
            count = 0

            for x in range(flood_cell_divide):
                x_coord = x + (x_block * flood_cell_divide)
                for y in range(flood_cell_divide):
                    y_coord = y + (y_block * flood_cell_divide)
                    if isWaterArr[y_coord][x_coord]:
                        total += heightmap[y_coord][x_coord]
                        count +=1

            avg_water_heights.append(0.0 if count == 0 else total / count)

    # Fill other cells using cells that have water
    has_changes = True
    while has_changes:
        has_changes = False

        for i in range(len(avg_water_heights)):
            if avg_water_heights[i] == 0.0:
                # Derive flood height using neighbouring cells
                neighbours = [avg_water_heights[i - flood_divide_ratio] if i >= flood_divide_ratio else 0.0,
                              avg_water_heights[i + flood_divide_ratio] if i < len(avg_water_heights) - flood_divide_ratio else 0.0,
                              avg_water_heights[i - 1] if i % flood_divide_ratio != 0 else 0.0,
                              avg_water_heights[i + 1] if (i+1) % flood_divide_ratio != 0 else 0.0]

                valid_cells = len([x for x in neighbours if x != 0.0])
                avg_water_heights[i] = sum(neighbours) / valid_cells if valid_cells != 0 else 0.0

                if avg_water_heights[i] != 0.0:
                    has_changes = True


    for y in range(len(output)):
        for x in range(len(output[y])):
            output[y][x] = avg_water_heights[discreteGridPosToID(x // flood_cell_divide,
                                                           y // flood_cell_divide, flood_divide_ratio)]

    im = Image.fromarray(output / max_h * 255.0).convert('RGB')
    im.save('output.png', optimize=True)

if __name__ == '__main__':
    main()