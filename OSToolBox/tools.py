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
    #read a text file with space separator
    #nomenclature style : <label name r g b>
    arr=np.loadtxt(nom_path,dtype='str')
    l=arr[:,0].astype('int')
    n=arr[:,1]
    c=arr[:,2:].astype('int')
    return l,n,c


"""PRINT"""
#to reset color from everywhere
RESETCOLOR = '\033[0m' #USAGE :: print(ost.RESETCOLOR,end="")

def PRINTCOLOR(r, g, b, background=False):
    #change the print color until you RESETCOLOR 
    #background=True for setting the mentionned color as background color
    #USAGE :: print(PRINTCOLOR(*color),end="")
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)


"""PLOT"""
def plot2dArray(a,wx=6,wy=6,save=None,show=1):
    #a=array
    #wx & wz = figure dimensions
    #save : path to save
    #show = true to display it
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
    #datalabel=name of data
    #xlabel & ylabel = name of x and Y data
    #save : path to save
    #show = true to display it
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
    #tps = reference time
    #USAGE :
        #t1=chrono()
        #do what you want
        #chrono(t1)
    if tps!=-1:
        print(time.perf_counter()-tps)
    return time.perf_counter()

"""PARSER"""
#argparse lib personnal methods
def str2bool(v):
    #convert input string to boolean or raise input error
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def SFTParser(v):
    #convert input string to boolean or raise input error
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

def SFParser(v):
    #try to convert input string to "false" bool else return string given
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

"""FILE"""
def getFileByExt(path, ext, rec=True,nat=True):
    #get all files which END WITH given string
    # rec : True if recursive search in subfolders
    # nat : True if sorting by natural order (1-2-10 and not 1-10-2)
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
    #get all files which CONTAIN given string
    # rec : True if recursive search in subfolders
    # nat : True if sorting by natural order (1-2-10 and not 1-10-2)
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
    #return root of file name
    #ex : pathBranch("media/ostocker/ilove.you") return "media/ostocker"
    return os.path.abspath(os.path.join(path, os.pardir))

def pathLeafExt(path):
    #return file name and ext
    #ex : pathBranch("media/ostocker/ilove.you") return "ilove.you"
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def pathLeaf(path):
    #return file name
    #ex : pathBranch("media/ostocker/ilove.you") return "ilove"
    return os.path.splitext(pathLeafExt(path))[0]

def pathExt(path):
    #return file ext
    #ex : pathBranch("media/ostocker/ilove.you") return ".tif"
    return os.path.splitext(pathLeafExt(path))[1]

def createDir(path):
    #create folder at path (with name) location skip and if already existing
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def checkNIncrementLeaf(path):
    #return the highest increment available on given path
    path_temp=path
    count=1
    while os.path.exists(path_temp):
        filename_temp=pathLeaf(path)+"_"+str(count)
        path_temp=pathBranch(path)+"/"+filename_temp+pathExt(path)
        count+=1
    return path_temp


def createDirIncremental(path,start=None):
    #create folder at path (with name) location and increment the name if already existing
    #start = give a int count starter
    #ex : [path("media/foo") for i in range(3)] wil create "media/foo_0","media/foo_1" & "media/foo_2"
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
    #get all Folder in path
    # return all dir and their given depth
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
    #take checkFoldersWithDepth output and sort it by dpeth
    #a=list of folders path
    #p = list of depth
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
    # create a random mask of pixel given a ratio (deprecated : only used in main_v0.py (main.py))
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
    # split an image into two given a mask (deprecated : only used in main_v0.py (main.py))
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
    #return array of boolean of dif between ground truth (gt) and prediction (pred) and ingore no_data_value
    #1=true, there is a diff, 0=no there is no diff
    dif=np.zeros(gt.shape)
    for i in range(dif.shape[0]):
        for j in range(dif.shape[1]):
            if pred[i,j]!=gt[i,j] and gt[i,j]!=no_data_value:
                dif[i,j]=1
    return dif

def slidingWindowsWithDrop(array, wX, wY, sX=None, sY=None):
    #can't rememeber (deprecated : only used in main_v0.py (main.py))
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

def cubify(arr, newshape):
    #from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    #from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

"""MATHS"""
def borderOutliers(a, lower=2, upper = 98):
    #return an array with clipped min and max value given lower and upper border
    result=np.zeros(a.shape)
    
    upper_quartile = np.percentile(a, upper)
    lower_quartile = np.percentile(a, lower)
    quartileSet = (lower_quartile, upper_quartile)
    
    temp=np.where(a >= quartileSet[0],a,quartileSet[0])
    result=np.where(temp <= quartileSet[1],temp,quartileSet[1])
    return result
    
def borderOutliers_v2(a, lower=2, upper = 98, force=False, quartileSet=(0,0) ):
    #return an array with clipped min and max value given lower and upper border
    #Force = True skip border finding and use quartile set
    #quartilSet : value to be forced (need force = True to be effective)
    if not force:
        upper_quartile = np.percentile(a, upper)
        lower_quartile = np.percentile(a, lower)
        quartileSet = (lower_quartile, upper_quartile)
    
    a=np.where(a >= quartileSet[0],a,quartileSet[0])
    
    return np.where(a <= quartileSet[1],a,quartileSet[1])

def getBorderOutliers(a, lower=2, upper = 98):
    #return border set of an array given lower and upper values
    upper_quartile = np.percentile(a, upper)
    lower_quartile = np.percentile(a, lower)
    quartileSet = (lower_quartile, upper_quartile)
    
    return quartileSet

def normalize(a,force=False,mini=0,maxi=0):
    # return normalized array a 
    #Force = True skip border finding and use mini and max values
    
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
