import PIL
import sys

from PIL import Image
import xarray as xr
import rioxarray as rio


# xarray Code from: https://help.marine.copernicus.eu/en/articles/5029956-how-to-convert-netcdf-to-geotiff

def main():
    # Get data
    nc_file = xr.open_dataset(sys.argv[1])

    # Extract Rasters
    raster_clay = nc_file['GLDAS_soilfraction_clay']
    raster_clay = raster_clay.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    raster_sand = nc_file['GLDAS_soilfraction_sand']
    raster_clay = raster_clay.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    raster_sand = raster_sand.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

    #Uncomment code if you want a specific location (in degrees)
    raster_clay = raster_clay.sel(lon=slice(30.0, 35.0), lat=slice(25.0, 35.0))
    raster_sand = raster_sand.sel(lon=slice(30.0, 35.0), lat=slice(25.0, 35.0))

    # Define projection
    raster_clay.rio.write_crs("epsg:4326", inplace=True)
    raster_sand.rio.write_crs("epsg:4326", inplace=True)

    # Write tiffs
    raster_clay.rio.to_raster('%s_clay.tiff' % sys.argv[2])
    raster_sand.rio.to_raster('%s_sand.tiff' % sys.argv[2])

if __name__ == '__main__':
    main()