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
import ntpath
import os
from osgeo import gdal,osr
import numpy as np
from collections import defaultdict
import argparse
import re
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


"""CLASSIFICATION"""
def readNomenclature(nom_path):
    #nomenclature style : <label name r g b>
    arr=np.loadtxt(nom_path,dtype='str')
    l=arr[:,0].astype('int')
    n=arr[:,1]
    c=arr[:,2:].astype('int')
    return l,n,c


"""PRINT"""
RESETCOLOR = '\033[0m'
def PRINTCOLOR(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)
    #USAGE :
#    print(get_color_escape(255, 128, 0) 
#      + get_color_escape(80, 30, 60, True)
#      + 'Fancy colors!' 
#      + RESET)

"""PLOT"""
def plot2dArray(a,wx=6,wy=6,save=None,show=1):
    fig = plt.figure(figsize=(wx, wy))
    
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(a)
    ax.set_aspect('equal')
    
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    
    
    ax1_divider = make_axes_locatable(ax)
    # add an axes to the right of the main axes.
    cax1 = ax1_divider.append_axes("right", size=0.1, pad="2%")
    plt.colorbar(orientation='vertical', cax=cax1)
    if save is not None :
        plt.savefig(save)
    if show : 
        plt.show()

def plotLogGraph(x,y,datalabel,xlabel,ylabel,title,save=None,show=1):
    fig,ax=plt.subplots()
    ax.semilogy(x, y,label=datalabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.legend()
    plt.grid(True)
    if save is not None :
        plt.savefig(save)
    if show : 
        plt.show()


"""TIMER"""
def chrono(tps=-1):
    if tps!=-1:
        print(time.perf_counter()-tps)
    return time.perf_counter()

"""PARSER"""
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def SFParser(v):
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

"""FILE"""
def getFileByExt(path, ext, rec=True,nat=True):
    list_file=[]
#    print(path)
    if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
    #                print(file)
                    if file.endswith(ext.lower()) or file.endswith(ext.upper()) :
                         list_file.append(os.path.join(root, file))
                if not rec:
                    break
#    print(list_file)
    if nat:
        list_file.sort(key=natural_keys)
    else :
        list_file.sort()
#    print(list_file)
    return list_file

def getFileBySubstr(path, substr, rec=True,nat=True):
    list_file=[]
#    print(path)
    if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
    #                print(file)
                    if substr.lower() in file or substr.upper() in file :
                         list_file.append(os.path.join(root, file))
                if not rec:
                    break
#    print(list_file)
    if nat:
        list_file.sort(key=natural_keys)
    else :
        list_file.sort()
#    print(list_file)
    return list_file

def pathBranch(path):
    return os.path.abspath(os.path.join(path, os.pardir))

def pathLeafExt(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def pathLeaf(path):
    return os.path.splitext(pathLeafExt(path))[0]

def pathExt(path):
    return os.path.splitext(pathLeafExt(path))[1]

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def checkNIncrementLeaf(path):
    path_temp=path
    count=1
    while os.path.exists(path_temp):
        filename_temp=pathLeaf(path)+"_"+str(count)
        path_temp=pathBranch(path)+"/"+filename_temp+pathExt(path)
        count+=1
    return path_temp


def createDirIncremental(path,start=None):
    path_temp=path
    count=1
    if start is not None :
        count = start
        path_temp = path_temp+"_"+str(count)
    while os.path.exists(path_temp):
        path_temp=path+"_"+str(count)
        count+=1
    createDir(path_temp)
    return path_temp

def checkFoldersWithDepth(directory, prof=0):
    dir_list = next(os.walk(directory))[1]
    a=[directory+"/"+n for n in dir_list]
    p=[prof for n in dir_list]
    prof+=1
    b=[]
    q=[]
    for d in dir_list:
#        print(d)
        b,q=checkFoldersWithDepth(directory +"/"+ d,prof)
        a+=b
        p+=q
    return a, p
#l=ost.sortFoldersByDepth(*ost.checkFoldersWithDepth(args.dir))
def sortFoldersByDepth(a,p):
    ind=sorted(range(len(p)), reverse=True, key=p.__getitem__)
#    print(ind)
    l=[]
    for i in ind:
        l.append(a[i])
    return l

"""LIST"""
#natural sorting algorithm
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    USE :
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def natural_sort_indices(temp):
    return [temp.index(i) for i in sorted(temp, key=lambda x: re.search(r"(\d+)\.py", x).group(1))]

"""GDAL RASTER """
def rasterCopy(raster_base, path, rasterType='GTiff', datatype=gdal.GDT_Int32, bands=-1,copy_nodata_values=True,margin=0):
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
            noDataValue=raster_base.GetRasterBand(b).GetNoDataValue()
            if noDataValue is not None :
                newRaster.GetRasterBand(b).SetNoDataValue(noDataValue)
    newRaster.SetProjection(proj)
    newRaster.SetGeoTransform(geoTransform)
    return newRaster

def array2raster(array, path, path_base, rasterType='GTiff', datatype=gdal.GDT_Float32, noDataValue=None, colors=None,margin=0):
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
    ds = gdal.Open(rasterfn)
    
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
#    print(cols,rows,bands)
    
    img_tensor=np.zeros((rows,cols,bands))
    for i in range(1,ds.RasterCount+1):
        band = ds.GetRasterBand(i)
        if normalizeAndBorder:
            img_tensor[:,:,i-1]=normalize(borderOutliers(band.ReadAsArray()))
        else:
            img_tensor[:,:,i-1] = band.ReadAsArray()
            
    if getBaseGeoTransform:
        #return tensor and geo info and epsg code
        prj = ds.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        return img_tensor, ds.GetGeoTransform(), int(srs.GetAttrValue("AUTHORITY", 1))
    else:
        return img_tensor
    
#def tensor2rasterManual(path, tensor,pixTL,gsd,epsg, rasterType='GTiff', datatype=gdal.GDT_Float32,noDataValue=None):
#    path=os.path.abspath(path)
#    # You need to get those values like you did.
#    bands=tensor.shape[1]
#    x_pixels = tensor.shape[2]  # number of pixels in x
#    y_pixels = tensor.shape[1]  # number of pixels in y
#    gsdX=gsd[0]
#    gsdY=gsd[1]
#    x_min = pixTL[0]
#    y_max = pixTL[1]  # x_min & y_max are like the "top left" corner.
#    proj = osr.SpatialReference()
#    proj.ImportFromEPSG( int(epsg) )
#    
#    driver = gdal.GetDriverByName(rasterType)
#    dataset = driver.Create(path,x_pixels,y_pixels,bands,datatype)
#    
#    dataset.SetGeoTransform((x_min,gsdX,0,y_max,0,gsdY))
#    dataset.SetProjection(proj.ExportToWkt())
#    
#    for b in range(bands
#    outBand=dataset.GetRasterBand(1)
#    outBand.WriteArray(array)
#    #nodata
#    if noDataValue is not None :
#       outBand.SetNoDataValue(noDataValue)
#    #colotable
#    if colors is not None :
#        cT = gdal.ColorTable()
#        # set color for each 
#        for i in range(len(colors)):
#            cT.SetColorEntry(i, colors[i])
#        if noDataValue is not None :
#            cT.SetColorEntry(noDataValue, (0,0,0,0))#nodata
#        # set color table and color interpretation
#        outBand.SetRasterColorTable(cT)
#        outBand.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
#    
#    dataset.FlushCache()  # Write to disk.
#    del outBand, dataset
#    return 0

"""ARRAY"""
def randomMaskCreator(array, ratio,no_data_value):
    mask=np.zeros(array.shape)
    dic=defaultdict(list) 
    dic_counter=defaultdict(int) 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            dic_counter[array[i,j]]+=1
            if ratio>np.random.random_sample() and array[i,j]!=no_data_value:
                mask[i,j]=1
                dic[array[i,j]].append((i,j)) 
    for key in dic:
        print("GT classe ",key,": ",len(dic[key]),"/",dic_counter[key])
    return mask,dic 

def applyMask(array,mask,no_data_value):
    positif=np.full(array.shape,no_data_value)
    negatif=np.full(array.shape,no_data_value)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
#            print(mask[i,j])
            if mask[i,j]==0:
                negatif[i,j]=array[i,j]
            elif mask[i,j]==1:
                positif[i,j]=array[i,j]
    return positif, negatif

def arrayDif(gt,pred,no_data_value):
    #1=true, tehre is a diff, 0=no there is no diff
    dif=np.zeros(gt.shape)
    for i in range(dif.shape[0]):
        for j in range(dif.shape[1]):
            if pred[i,j]!=gt[i,j] and gt[i,j]!=no_data_value:
                dif[i,j]=1
    return dif

def slidingWindowsWithDrop(array, wX, wY, sX=None, sY=None):
    sX = wX if sX is None else sX
    sY = wY if sY is None else sX
    stepX = int((array.shape[0]-wX)/sX +1)
    stepY = int((array.shape[1]-wY)/sY +1)
    data=[]
    for j in range(stepY):
        for i in range (stepX):
            data.append(array[i*sX:i*sX+wX, j*sY:j*sY+wY])
    return data

"""TENSOR"""
def tensor2raster(tensor, path, path_base, format='GTiff', datatype=gdal.GDT_Int32):
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

def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

"""MATHS"""
def borderOutliers(a, lower=2, upper = 98):
    result=np.zeros(a.shape)
    
    upper_quartile = np.percentile(a, upper)
    lower_quartile = np.percentile(a, lower)
    quartileSet = (lower_quartile, upper_quartile)
    
    temp=np.where(a >= quartileSet[0],a,quartileSet[0])
    result=np.where(temp <= quartileSet[1],temp,quartileSet[1])
    return result
    
def borderOutliers_v2(a, lower=2, upper = 98, force=False, quartileSet=(0,0) ):
    if not force:
        upper_quartile = np.percentile(a, upper)
        lower_quartile = np.percentile(a, lower)
        quartileSet = (lower_quartile, upper_quartile)
    
    a=np.where(a >= quartileSet[0],a,quartileSet[0])
    
    return np.where(a <= quartileSet[1],a,quartileSet[1])

def getBorderOutliers(a, lower=2, upper = 98):
    upper_quartile = np.percentile(a, upper)
    lower_quartile = np.percentile(a, lower)
    quartileSet = (lower_quartile, upper_quartile)

    return quartileSet

def normalize(a,force=False,mini=0,maxi=0):
    if not force:
        maxi=a.max()
        mini=a.min()
    return (a-mini)/(maxi-mini)

""" STRING"""
def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)
   
def find_all(string,substring):
    return [m.start() for m in re.finditer(substring,string)]
