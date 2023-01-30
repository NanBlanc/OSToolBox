#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Tue May 28 10:40:24 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OSToolBox    
"""

import os
from osgeo import gdal,osr
import numpy as np
import OSToolBox as ost



"""GDAL RASTER """
def rasterCopy(raster_base, path, rasterType='GTiff', datatype=gdal.GDT_Int32, bands=-1,copy_nodata_values=True,margin=0):
    #return new raster
    #raster_base = raster to copy
    #path = new raster path
    #bands=number of band to create if int(-1) : copy raster_path band number
    #margin = reduce of size beetween new raster and raster 
    geoTransform = raster_base.GetGeoTransform()
    if margin!=0:
        geoTransform=list(geoTransform)
        geoTransform[0] = geoTransform[0]+margin*geoTransform[1]
        geoTransform[3] = geoTransform[3]+margin*geoTransform[5]

    # buffering to handle edge issues
    cols = raster_base.RasterXSize-2*margin
    rows = raster_base.RasterYSize-2*margin
    proj = raster_base.GetProjection()
    if bands ==-1:
        bands=raster_base.RasterCount
    
    driver = gdal.GetDriverByName(rasterType)
    newRaster = driver.Create(path, cols, rows, bands, datatype)
    if copy_nodata_values:
        for b in range(1,bands+1):
            try:
                noDataValue=raster_base.GetRasterBand(b).GetNoDataValue()
                if noDataValue is not None :
                    newRaster.GetRasterBand(b).SetNoDataValue(noDataValue)
            except:
                pass
    newRaster.SetProjection(proj)
    newRaster.SetGeoTransform(geoTransform)
    return newRaster

def array2raster(array, path, path_base, rasterType='GTiff', datatype=gdal.GDT_Float32, noDataValue=None, colors=None,margin=0):
    #create a new image of an array by copying path_base metadata
    #array (2D) to create 
    #path_base = path to image to copy (will be based one first band)
    #path = new image path
    #margin = reduce of size beetween new raster and raster
    #color : color list in "r,g,b" may be deprecated
    path_base=os.path.abspath(path_base)
    raster_base = gdal.Open(path_base)
    path=os.path.abspath(path)
    new_raster=rasterCopy(raster_base, path, rasterType=rasterType, datatype=datatype ,bands=1,margin=margin)
    outBand=new_raster.GetRasterBand(1)
    #nodata
    if noDataValue is not None :
        outBand.SetNoDataValue(noDataValue)
    #colotable
    if colors is not None :
        cT = gdal.ColorTable()
        # set color for each 
        for i in range(len(colors)):
            cT.SetColorEntry(i, colors[i])
        if noDataValue is not None :
            cT.SetColorEntry(noDataValue, (0,0,0,0))#nodata
        # set color table and color interpretation
        outBand.SetRasterColorTable(cT)
        outBand.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    outBand.WriteArray(array)
    new_raster.FlushCache()
    del outBand, new_raster
    return 0

def array2rasterManuel(path, array,pixTL,gsd,epsg, rasterType='GTiff', datatype=gdal.GDT_Float32,noDataValue=None, colors=None):
    #create a new image of an array with given metadata
    #array (2D) to create 
    #path = new image path
    #pixTL=tuple or list of with top left pixel coordinate (x,y)
    #gsd=tuple or list of with ground sampling distance (x,y)
    #epsg = code of coordiante system
    #color : color list in "r,g,b" may be deprecated
    path=os.path.abspath(path)
    # You need to get those values like you did.
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    gsdX=gsd[0]
    gsdY=gsd[1]
    x_min = pixTL[0]
    y_max = pixTL[1]  # x_min & y_max are like the "top left" corner.
    proj = osr.SpatialReference()
    proj.ImportFromEPSG( int(epsg) )
    
    driver = gdal.GetDriverByName(rasterType)
    dataset = driver.Create(path,x_pixels,y_pixels,1,datatype)
    
    dataset.SetGeoTransform((x_min,gsdX,0,y_max,0,gsdY))
    dataset.SetProjection(proj.ExportToWkt())
    
    outBand=dataset.GetRasterBand(1)
    outBand.WriteArray(array)
    #nodata
    if noDataValue is not None :
       outBand.SetNoDataValue(noDataValue)
    #colotable
    if colors is not None :
        cT = gdal.ColorTable()
        # set color for each 
        for i in range(len(colors)):
            cT.SetColorEntry(i, colors[i])
        if noDataValue is not None :
            cT.SetColorEntry(noDataValue, (0,0,0,0))#nodata
        # set color table and color interpretation
        outBand.SetRasterColorTable(cT)
        outBand.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    dataset.FlushCache()  # Write to disk.
    del outBand, dataset
    return 0

def raster2tensor(rasterfn,normalizeAndBorder=False,getBaseGeoTransform=False):
    #read an multi band image and convert it into 3D matrix
    ds = gdal.Open(rasterfn)
    
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
#    print(cols,rows,bands)
    
    img_tensor=np.zeros((rows,cols,bands))
    for i in range(1,ds.RasterCount+1):
        band = ds.GetRasterBand(i)
        if normalizeAndBorder:
            img_tensor[:,:,i-1]=ost.normalize(ost.borderOutliers(band.ReadAsArray()))
        else:
            img_tensor[:,:,i-1] = band.ReadAsArray()
            
    if getBaseGeoTransform:
        #return tensor and geo info and epsg code
        prj = ds.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        return img_tensor, ds.GetGeoTransform(), int(srs.GetAttrValue("AUTHORITY", 1))
    else:
        return img_tensor


"""TENSOR"""
def tensor2raster(tensor, path, path_base, format='GTiff', datatype=gdal.GDT_Int32):
    #create a new image of an tensor (3D) by copying path_base metadata
    #numpy matrix (3D) to create 
    #path_base = path to image to copy (will be based one first band)
    #path = new image path
    path_base=os.path.abspath(path_base)
    raster_base = gdal.Open(path_base)
    path=os.path.abspath(path)
    print(tensor.shape)
    new_raster=rasterCopy(raster_base, path, bands=tensor.shape[2])
    for i in range (tensor.shape[2]):
        outBand=new_raster.GetRasterBand(i+1)
        outBand.WriteArray(tensor[:,:,i])
    new_raster=None 
    return 0