#!/usr/bin/python

# hgt to png
# Originally by Jonas Otto - helloworld@jonasotto.de
# based on information found here https://gis.stackexchange.com/questions/43743/how-to-extract-elevation-from-hgt-file
# and the official SRTM Documentation https://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
import os
import sys
import struct
from PIL import Image

def map(val, in_min, in_max, out_min, out_max ):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def usage():
    print(".hgt to png converter usage: python hgt2png.py <inputfile.hgt>  <resolution 1 or 3 standard is 1>")

def main (argv):
    print (argv)
    data_resolution = 1
    image_resolution = 3601
    height_root_dir = argv[0]

    min_value = 1000000000000000000000.0
    max_value = 0.0

    void_value = -32768

    raw_images = {}

    for root, dirs, files in os.walk(height_root_dir):
        for file in files:
            if not file.endswith('.hgt'):
                continue
            height_map_data_ordered = []
            void_counter = 0
            print("Processing file %s" % file)
            with open(os.path.join(root, file), "rb") as source_file:
                for y in range(image_resolution):
                    for x in range(image_resolution):
                        index = (( y * image_resolution) + x)*2
                        source_file.seek( index )
                        buf = source_file.read(2)
                        value = struct.unpack('>h', buf)[0]
                        if value != void_value:
                            height_map_data_ordered.append(value)
                            if value > max_value:
                                max_value = value
                            elif value < min_value:
                                min_value = value
                        else:
                            void_counter += 1
                            height_map_data_ordered.append(0)
                            if 0 < min_value:
                                min_value = 0

            raw_images[file[:-4]] = height_map_data_ordered

    print ("final min and max values are: ", min_value, max_value)
    #print ("map data to greyscale")
    #for key in raw_images:
        #height_map = Image.new('RGB', (image_resolution, image_resolution))
        #height_map_data = []

        #for value in raw_images[key]:
            #mapped_value = int(round(map(value,min_value,max_value,0,255)))
            #height_map_data.append((mapped_value,mapped_value,mapped_value))

        #print ("drawing Heightmap")
        #height_map.putdata(height_map_data)
        #print ("saving Heightmap")
        #height_map.save("%s.png" % key)
        #print ("done")

if __name__ == "__main__":
    main(sys.argv[1:])