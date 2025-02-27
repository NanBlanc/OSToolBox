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
import numpy as np
from collections import defaultdict
import argparse
import re
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import sys
import warnings


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

def plotGraph(x,array_y,datalabel="",xlabel="",ylabel="",xlimit=None,ylimit=None,title="",fontsize=10,log=False,save=None,grid=True,dpi=600,show=1):
    #array_y can be y data or an array of y datas to plot together
    #datalabel=name of data
    #xlabel & ylabel = name of x and Y data
    #save : path to save
    #show = true to display it
    plt.rcParams.update({'font.size': fontsize})
    fig,ax=plt.subplots()
    array_y=np.asarray(array_y)#cast as np array
    if len(array_y.shape)>1:
        for i,y in enumerate(array_y):
            if log:
                ax.semilogy(x, y,label=datalabel[i])
            else:
                ax.plot(x, y,label=datalabel[i])
    else :
        if log:
            ax.semilogy(x, array_y,label=datalabel)
        else:
            ax.plot(x, array_y,label=datalabel)
    if xlimit is not None : 
        plt.xlim(xlimit)
    if ylimit is not None :
        plt.ylim(ylimit)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.legend()
    plt.grid(grid)
    plt.tight_layout()
    if save is not None :
        plt.savefig(save,dpi=dpi)
    if show : 
        plt.show()


"""TIMER"""
def chrono(temps=-1):
    #tps = reference time
    #USAGE :
        #t1=chrono()
        #do what you want
        #chrono(t1)
    if temps!=-1:
        print(time.perf_counter()-temps)
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
def countFiles(directory):
     file_count = sum(len(files) for _, _, files in os.walk(directory))
     return file_count

def getFileByExt(path, ext, rec=True,nat=True, caseSensitive=False):
    #get all files which END WITH given string
    # rec : True if recursive search in subfolders
    # nat : True if sorting by natural order (1-2-10 and not 1-10-2)
    list_file=[]
#    print(path)
    if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    # print(file)
                    if caseSensitive:
                        if file.endswith(ext):
                             list_file.append(os.path.join(root, file))  
                    else:                          
                        if file.lower().endswith(ext.lower()):
                             list_file.append(os.path.join(root, file))
                if not rec:
                    break
#    print(list_file)
    if nat:
        list_file.sort(key=_naturalKeys)
    else :
        list_file.sort()
#    print(list_file)
    return list_file

def getFileBySubstr(path, substr, rec=True,nat=True,caseSensitive=False):
    #get all files which CONTAIN given string
    # rec : True if recursive search in subfolders
    # nat : True if sorting by natural order (1-2-10 and not 1-10-2)
    list_file=[]
#    print(path)
    if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if caseSensitive:
                        if substr in file:
                             list_file.append(os.path.join(root, file))
                    else:
                        if substr.lower() in file.lower():
                             list_file.append(os.path.join(root, file))
                if not rec:
                    break
#    print(list_file)
    if nat:
        list_file.sort(key=_naturalKeys)
    else :
        list_file.sort()
#    print(list_file)
    return list_file

def getDirBySubstr(path, substr, rec=True,nat=True,caseSensitive=False):
    #get all dir paths
    # rec : True if recursive search in subfolders
    # nat : True if sorting by natural order (1-2-10 and not 1-10-2)
    list_dir=[]
#    print(path)
    if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for dire in dirs:
                    if caseSensitive:
                        if substr in dire:
                             list_dir.append(os.path.join(root, dire))
                    else:
                        if substr.lower() in dire.lower():
                             list_dir.append(os.path.join(root, dire))
                if not rec:
                    break
#    print(list_file)
    if nat:
        list_dir.sort(key=_naturalKeys)
    else :
        list_dir.sort()
#    print(list_file)
    return list_dir

def pathBranch(path,n=1):
    #return root of file name
    #ex : pathBranch("media/ostocker/ilove.you") return "media/ostocker"
    #n : number of back setps : pathBranch("media/ostocker/ilove.you",2) return "media"
    for i in range(n):
        path=os.path.abspath(os.path.join(path, os.pardir))
    return path

def pathRelative(path,n):
    #return rrelative path give certain depth n
    #ex : pathBranch("media/ostocker/ilove.you",1) return "ostocker/ilove.you"
    leafed_branch=pathLeafExt(path)
    rooted_branch=pathBranch(path)
    for i in range(n):
        leafed_branch=pathLeafExt(rooted_branch)+"/"+leafed_branch
        rooted_branch=pathBranch(rooted_branch)
    return leafed_branch

def pathLeafExt(path):
    #return file name and ext
    #ex : pathBranch("media/ostocker/ilove.you") return "ilove.you"
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def pathLeaf(path):
    #return file name
    #ex : pathBranch("media/ostocker/ilove.you") return "ilove"
    return os.path.splitext(pathLeafExt(path))[0]

def pathBranchLeaf(path):
    #return file name
    #ex : pathBranch("media/ostocker/ilove.you") return "media/ostocker/ilove"
    return pathBranch(path)+"/"+pathLeaf(path)

def pathLeafStringAppend(path,s):
    #return file name
    #append a string before extension
    #ex : pathBranch("media/ostocker/ilove.you","_me") return "media/ostocker/ilove_me.you"
    return pathBranch(path)+"/"+pathLeaf(path)+s+pathExt(path)

def pathExt(path):
    #return file ext
    #ex : pathBranch("media/ostocker/ilove.you") return ".tif"
    return os.path.splitext(pathLeafExt(path))[1]

def createDir(path):
    #create folder at path (with name) location skip and if already existing
    os.makedirs(path, exist_ok=True)
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
def _atoi(text):
    return int(text) if text.isdigit() else text

def _naturalKeys(text):
    '''
    USE :
    alist.sort(key=_naturalKeys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ _atoi(c) for c in re.split(r'(\d+)', text) ]

def naturalSortIndices(temp):
    return [temp.index(i) for i in sorted(temp, key=lambda x: re.search(r"(\d+)\.py", x).group(1))]


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

def movingAverage(x, window_size):
    #For smoothing curve, it does loop around, so return a vector of the same size
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(x, window, 'same')

""" STRING"""
def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)
   
def find_all(string,substring):
    return [m.start() for m in re.finditer(substring,string)]


def stringlist_to_list(culotte,sep=" "):
    #take a list written in strings : "['ab', 'pm']" and return it as a real list ['ab', 'pm']
    short=culotte[1:-1].split(sep=sep)
    l=[]
    for s in short:
        if s[0]=="\"" or s[0]=="'":
            l.append(s[1:-1])
        else :
            l.append(float(s))
    return l


def intInString(string):
    #get_int_in_string("/home/parcelles1/sub_21/paclot_887","_") => [1,21,887]
    return np.array([int(s) for s in re.findall('[0-9]+', string)])

""" POINT CLOUD TRANSFORM FUNCTIONS """
def rotationZ(points, angle, degree=False):
    """
    apply rotation of angle around Z axis 
    
    points : 2d np array where XYZ should be 3 first column
    angle : angle of rotation in radian
    degree(False) : True if you want to use angles in degree
    """
    angle=angle*np.pi/180 if degree else angle
    rot_mat = np.array([[np.cos(angle),-np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], 
                        [0, 0, 1]])
    points[:, :3] = np.dot(points[:, :3], rot_mat)
    return points

def rotationXYZ(points, angles, degree=False, inverse=False):
    """
    apply rotation of angle around X then Y then Z axis of angle in angles
    
    points : 2d np array where XYZ should be 3 first column
    angles : list of angle of rotation in radian for X, Y and Z
    inverse(False) : True if you want to apply the rotation order as Z then Y then X axis
    """
    angles=angles*np.pi/180 if degree else angles
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rx, np.dot(Ry,Rz)) if inverse else np.dot(Rz, np.dot(Ry,Rx)) 
    points[:, :3] = np.dot(points[:, :3], R)
    return points

def featureAugmentation(points,column,maximum=None,sigma=None):
    """
    apply gaussian noise then normalize a list a values given a maximum
    
    values : array of values (needs shape attribute)
    column : index of column of which to transform is applied
    maximum(None) : max value used to normalise. If none take max of values
    sigma(None) : sigma of gaussian distrubution. If none take 1% of maximum
    """
    values=points[:,column]
    #Value jittering and Normalisation given the maximum (default sigma = 1percent of max)
    maximum = maximum if maximum is not None else np.max(values)
    sigma=maximum/100 if sigma is None else sigma
    clip=sigma*4
    values += np.clip(sigma * np.random.randn(*values.shape), -1*clip, clip)
    points[:,column]=values/maximum
    return points

def cuboidDrop(points,cuboid_size):
    """
    remove points located in a cube of dimension size^3 and random orientation centered on a random point of points
    
    points : 2d np array where XYZ should be 3 first column
    cuboid_size : size of the cube : size*size*size
    """
    drop_center = points[np.random.choice(points.shape[0]), :3].copy()#ATTENTION DROP CENTER NEEDS TO BE COPIED OTHERWISE LOST DURING TRANSLATION
    #define drop_center as coordinate center
    points[:,:3]-=drop_center
    
    #do random rotation
    angles = np.random.randn(3)*2*np.pi 
    points[:,:3]=rotationXYZ(points[:,:3],angles)
    
    max_xyz = cuboid_size/2
    min_xyz = -cuboid_size/2
    
    upper_idx = np.sum((points[:,:3] < max_xyz).astype(np.int32), 1) == 3
    lower_idx = np.sum((points[:,:3] > min_xyz).astype(np.int32), 1) == 3

    new_pointidx = ~((upper_idx) & (lower_idx))
    points=points[new_pointidx,...]
    
    #inverse rotation and translation
    points[:,:3]=rotationXYZ(points[:,:3],-angles,inverse=True)
    points[:,:3]+=drop_center
    return points

def jittering(points,sigma=0.05, clip=None):
    """
    move points on the 3 axis given a gaussian probability of sigma, clipped at clip
    
    points : 2d np array where XYZ should be 3 first column
    sigma(0.05) : sigma of the distance normal distribution of XYZ movement
    clip(None) : maximum/minimum value of distance from gaussian function. if None max/min = 4*sigma 
    """
    clip=4*sigma if clip is None else clip
    points[:,:3] += np.clip(sigma * np.random.randn(points.shape[0],3), -1*clip, clip)    
    return points
    
    
def randomCuboidDrop(points, min_dropped, max_dropped, size,sigma):
    """
    Repeat in a uniform distribution probabilty between min_dropped and max_dropped the removal of points located in a cube of dimension distrubtion gaussian of sigma=sigma(clipped at 4*sigma) and mu=size^3 and random orientation centered on a random point of points
    
    points : 2d np array where XYZ should be 3 first column
    min_dropped : minimum number of cube dropped
    min_dropped : maximum number of cube dropped
    cuboid_size : center of the size normal distribution of cubes : size*size*size
    sigma : sigma of the size normal distribution of cubes
    """
    sizes=np.random.randn(int(np.random.uniform(min_dropped,max_dropped)))*sigma+size
    sizes=np.clip(sizes,-4*sigma+size,4*sigma+size)
    for s in sizes:
        points=cuboidDrop(points,s)
    return points

def randomFlip(points,flip_x=True,flip_y=True,flip_z=False):
    """
    applies a flip (inversion the coordinates) with a 0.5 probability on choosen axis 
    
    points : 2d np array where XYZ should be 3 first column
    flip_x(True) : if True inverse x coordinate 50% of the time 
    flip_y(True) : if True inverse y coordinate 50% of the time 
    flip_z(False) : if True inverse z coordinate 50% of the time 
    """
    if np.random.random() > 0.5 and flip_x:
        # print("flipped X")
        points[:,0,...] = -1 * points[:,0,...]
    if np.random.random() > 0.5 and flip_y:
        # print("flipped Y")
        points[:,1,...] = -1 * points[:,1,...]
    if np.random.random() > 0.5 and flip_z:
        # print("flipped Z")
        points[:,2,...] = -1 * points[:,2,...]
    return points

def randomDrop(points, max_dropped_ratio=0.5):
    """
    applies randomly drop a ratio of points in points where ratio is uniform value between 0 and max_dropped_ratio
    
    points : 2d np array where XYZ should be 3 first column
    max_dropped_ratio(0.5) : maximum ratio selected of points to drop
    """
    dropout_ratio =  np.random.random()*max_dropped_ratio # 0~0.875
    drop_idx = np.where(np.random.random((points.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        # points[drop_idx,:] = points[0,:] # set to the first point (ORIGINAL VERSION WHERE THE REPLACE WITH FIRST INSTANCE, TO KEEP SAME NB OF POINTS)
        points=np.delete(points,drop_idx,0)
    return points

def qcsfTransform(points,drop_ratio=0.1,min_cube_drop=2,max_cube_drop=6,cube_size=4,sigma_cube_size=1,sigma_jittering=0.05,max_intensity=1025):
    ##point drop
    #random point drop
    points=randomDrop(points,drop_ratio)
    #cuboid drop
    points=randomCuboidDrop(points,min_cube_drop,max_cube_drop,cube_size,sigma_cube_size)
    
    ##position
    # translation :
    translation_xy=np.random.uniform(-20,0,2)
    translation_xyz=np.append(translation_xy,np.random.uniform(-10,10))
    points[:,:3]+=translation_xyz
    #scene flip
    points=randomFlip(points)
    # rotation :
    theta = np.random.uniform(0, 2*np.pi)
    points=rotationZ(points, theta)
    # jittering : 
    points=jittering(points,sigma_jittering)
    
    ##features
    #intensity augment
    points=featureAugmentation(points,3,max_intensity)
    return points

""" PLY FORMAT POINT CLOUDS"""
# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'int64', 'i8'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


def _parseHeader(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            if b'face' in line:
                raise ValueError('Trying to read a mesh : use arg triangular_mesh=True')
                
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            if line[1] not in ply_dtypes.keys():
                raise ValueError('Unsupported faces property : ' + ' '.join(str(l) for l in line) +'\n !!!!!!!!!! /!\\ Please Contact Olivier /!\\')
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
        


    return num_points, properties


def _parseMeshHeader(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                if line[1] not in ply_dtypes.keys():
                    raise ValueError('Unsupported point property format : ' + str(line[1]) +'\n !!!!!!!!!! /!\\ Please Contact Olivier /!\\')
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'face':
                line = line.split()
                if not(line[2]==b'uchar' or line[2]==b'uint8') or not(line[3]==b'int' or line[3]==b'int32'):
                    raise ValueError('Unsupported face properties : ' + ' '.join(l.decode('utf-8') for l in line) +'\n !!!!!!!!!! /!\\ Please Contact Olivier /!\\')

    return num_points, num_faces, vertex_properties


def readPly(filename, fields_names=False, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result 
    np.array of points + features
    +
    [if triangular_mesh=True] np.array of faces
    +
    [if return_fields_names=True]: np.array of features fields names
    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('Is it .PLY Format ?\nThe file does not start whith the word "ply"')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = _parseMeshHeader(plyfile, ext)
            
            vertex = 0
            faces = 0
            if fmt == "ascii":
                #split the long list outputed by fromfile at the right position between vertex and faces
                data = np.split(np.fromfile(plyfile,sep=" "),[num_points*len(properties)])
                #reshape the data
                vertex=data[0].reshape((num_points,len(properties)))
                try :
                    full_faces=data[1].reshape((num_faces,4))
                except:
                    raise ValueError('This file is not a triangular mesh (only 3 vertex per face)')
                faces=full_faces[:,1:]

            else :
                # Get point data
                vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
                vertex_data=vertex_data.astype([('', '<f8')]*len(properties))
                vertex=vertex_data.view('<f8').reshape(vertex_data.shape + (-1,))
    
                # Get face data
                face_properties = [('k', ext + 'u1'),
                                   ('v1', ext + 'i4'),
                                   ('v2', ext + 'i4'),
                                   ('v3', ext + 'i4')]
                faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
                
                # Return vertex data and concatenated faces
                faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            if fields_names:
                return vertex, faces, [p[0] for p in properties]
            else :
                return vertex, faces

        else:
            # Parse header
            num_points, properties = _parseHeader(plyfile, ext)                
            # Get data
            data=0
            if fmt == "ascii":
                data = np.fromfile(plyfile,sep=" ").reshape((num_points,len(properties)))                
            else :
                data = np.fromfile(plyfile, dtype=properties, count=num_points)
                #convert to np.ndarray, unstructured, in float64
                data=data.astype([(properties[i][0], '<f8') for i in range(len(properties))])
                data=data.view('<f8').reshape(data.shape + (-1,))
            if fields_names:
                return data, [p[0] for p in properties]
            else :
                return data


def _headerProperties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def writePly(filename, field_list, field_names, storage="binary", comments=None, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 
        Example : two points x=1,y=2,z=3 and x=1,y=2,z=4
            YES : field_list=np.array([[1,2,3],[1,2,4]])
            NO : field_list=[[1,2,3],[1,2,4]]
            YES : field_list=[[1,1],[2,2],[3,4]]
            NO : field_list=np.array([[1,1],[2,2],[3,4]])

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.`
    
    comments : list
        every comment you want to add, each list entry is an other line
    
    triangular_faces : list, numpy array (2 dimension : faces, indexes)
        list of list of 3 vertex indexes that share a common face. 

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> comments = ['offsets 123 545 112', 'generated by me the great']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form (be a list if alone or tuple)
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    
    #convert to np.array if not
    for i, field in enumerate(field_list):
        if not isinstance(field,np.ndarray):
            field_list[i]=np.array(field)
    
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    
    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions :', [field.shape for field in field_list])
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print(n_fields,field_names)
        print('wrong number of field names')
        return False
    
    #Check if no int 64
    n=0
    for i, field in enumerate(field_list):
        if field_list[i].dtype.name == "int64":
            field_list[i]=field_list[i].astype("int32")
            warnings.warn("OST WARNING : recasted field {} from int64 to int32, cause int64 is not accepted for plyfile. To remove this warning, recast yourself (to float or int32)".format(field_names[n]))
        n+=field_list[i].shape[1]
    

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        if storage=="binary":
            header.append('format binary_' + sys.byteorder + '_endian 1.0')
        elif storage=="ascii":
            header.append('format ascii 1.0')
        else:
            raise ValueError('Unsupported file format : ' + str(storage) + ', select "binary" or "ascii"')
            
        if comments is not None:
            if isinstance(comments, str) :
                header.append("comment " + comments)
            else :
                for c in comments :
                    header.append("comment " + c)

        # Points properties description
        header.extend(_headerProperties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            triangular_faces = np.array(triangular_faces)
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all header lines
        for line in header:
            plyfile.write("%s\n" % line)
    
    
    #Writting
    if storage=="ascii":
        # open in binary/append to use tofile
        with open(filename, 'a') as plyfile:
            data= np.hstack(field_list)
            savetxtCompact(plyfile, data,' ')

            if triangular_faces is not None:
                k=np.full((triangular_faces.shape[0],1), 3)
                data=np.hstack([k,triangular_faces],)
                np.savetxt(plyfile, data, '%i',' ')
                
    elif storage=="binary":
        with open(filename, 'ab') as plyfile:
            i = 0
            type_list = []
            for fields in field_list:
                for field in fields.T:
                    type_list += [(field_names[i], field.dtype.str)]
                    i += 1
            data = np.empty(field_list[0].shape[0], dtype=type_list)
            i = 0
            for fields in field_list:
                for field in fields.T:
                    data[field_names[i]] = field
                    i += 1

            data.tofile(plyfile)

            if triangular_faces is not None:
                triangular_faces = triangular_faces.astype(np.int32)
                type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
                data = np.empty(triangular_faces.shape[0], dtype=type_list)
                data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
                data['0'] = triangular_faces[:, 0]
                data['1'] = triangular_faces[:, 1]
                data['2'] = triangular_faces[:, 2]
                data.tofile(plyfile)
    return True

""" TEXT FUNCTIONS """

def savetxtCompact(fname, x, delimiter=','):
    #from https://stackoverflow.com/questions/24691755/how-to-format-in-numpy-savetxt-such-that-zeros-are-saved-only-as-0
    #Check if fname is path or file handle
    opened=False
    if not(hasattr(fname, 'read') and hasattr(fname, 'write')):
        fh=open(fname, 'w')
        opened=True
    else :
        fh=fname
    #writting loop
    for row in x:
        line = delimiter.join(str(np.float64(value)).rstrip('0').rstrip('.') if not "e+" in str(np.float64(value)) else str(np.float64(value)) for value in row)
        if "e+" in line :
            warnings.warn('OST WARNING : Values too big for memory precision, forced to use scientific notation')
        fh.write(line + '\n')
    #close file handle if opened here
    if opened:
        fh.close()
        

""" AMAPVOXEL FUNCTIONS """
#function edited from hannah weiser opaque voxel article
def save_amap_vox(fname, vox_list, max_corner_center, min_corner_center, resolution_vox, pad_list=None, transmittance_list=None):
    vox_list=np.array(vox_list)
    max_corner_center=np.array(max_corner_center)
    min_corner_center=np.array(min_corner_center)
    # print("writing vox...")
    with open(fname, "w") as outfile:
        outfile.write("VOXEL SPACE\n")
        outfile.write("#min_corner: %f %f %f\n" % (min_corner_center[0], min_corner_center[1], min_corner_center[2]))
        outfile.write("#max_corner: %f %f %f\n" % (max_corner_center[0], max_corner_center[1], max_corner_center[2]))
        split=max_corner_center-min_corner_center
        outfile.write("#split: %i %i %i\n" % (int(split[0]/resolution_vox), int(split[1]/resolution_vox), int(split[2]/resolution_vox)))
        outfile.write("#res: %f\n" % resolution_vox)
        outfile.write("i j k PadBVTotal angleMean bsEntering bsIntercepted bsPotential ground_distance lMeanTotal lgTotal nbEchos nbSampling transmittance attenuation attenuationBiasCorrection\n")
        #loop per voxel
        arr = np.zeros((vox_list.shape[0], 16))
        for idx, x in enumerate(vox_list):
            i, j, k = x
            arr[idx, :3] = [i, j, k]
        if pad_list is not None:
            arr[:, 3] = pad_list
        if transmittance_list is not None:
            arr[:, 3] = pad_list
        
        #save
        np.savetxt(outfile, arr, delimiter=" ", fmt="%i %i %i %f %i %i %i %i %i %i %i %i %i %f %i %i")
        
        
        
""" METRICS SCORE """
from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    #class to manage confusion matrix and compute associated metrics
    def __init__(self, n_class, class_names,noDataValue):
        self.CM=np.zeros((n_class,n_class))
        self.n_class=n_class
        self.class_names=class_names
        self.noDataValue=noDataValue
        
    def clear(self):
        self.CM=np.zeros((self.n_class,self.n_class))
    
    def add_batch(self, gt, pred):
        labeled= gt!=self.noDataValue
        if labeled.any():
            self.CM+=confusion_matrix(gt[labeled], pred[labeled], labels = list(range(0,self.n_class)))
    
    def add_matrix(self,matrix):
        self.CM+=matrix
        
    def overall_accuracy(self):
        return 100*self.CM.trace()/self.CM.sum()
    
    def class_IoU(self, show=1):
        #RETURN : miou & list of ious
        #show = True for printing IoU
        ious = np.full(self.n_class, 0.)
        for i_class in range(self.n_class):
            diviseur=(self.CM[i_class,:].sum()+self.CM[:,i_class].sum()-self.CM[i_class,i_class])
            if diviseur ==0:
#                print("WAS ZERO")
                ious[i_class]=np.nan
            else:
#                print("WAS NOT ZERO")
                ious[i_class] = self.CM[i_class,i_class] /diviseur
        if show:
            print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
        return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum(), ious
    
    def printPerf(self,outDir):
        #calculate and output score
        np.savetxt(outDir+"/precision.txt",self.class_precision(),fmt="%.4f")
        np.savetxt(outDir+"/recall.txt",self.class_recall(),fmt="%.4f")
        np.savetxt(outDir+"/f1_score.txt",self.class_f1_score(),fmt="%.4f")
        np.savetxt(outDir+"/oa.txt",[self.overall_accuracy()],fmt="%.4f")
        #calculate & output IoU and mIoU
        mious=self.class_IoU()
        np.savetxt(outDir+"/miou.txt",[mious[0]],fmt="%.4f")
        np.savetxt(outDir+"/iou.txt",mious[1]*100,fmt="%.4f")
        #output CM
        np.savetxt(outDir+"/cm.txt",self.CM,fmt="%d")
        
    def class_precision(self):
        self.precision=[row[i]/sum(row)*100 for i,row in enumerate(self.CM)]
        return self.precision
    
    def class_recall(self):
        self.recall=[row[i]/sum(row)*100 for i,row in enumerate(self.CM.T)]
        return self.recall
    
    def class_f1_score(self):
        self.class_precision()
        self.class_recall()
        self.f1_score=[2*p*r/(p+r) for (p,r) in zip(self.precision,self.recall)]
        return self.f1_score


###################################################
###################################################
""" OLD NAMING CONVENTION """

def write_ply(filename, field_list, field_names, storage="binary", comments=None, triangular_faces=None):
    # print("DEPRECATED in 1.7, please use writePly() function")
    return writePly(filename, field_list, field_names, storage, comments, triangular_faces)
def read_ply(filename, triangular_mesh=False):
    # print("DEPRECATED in 1.7, please use readPly() function")
    return readPly(filename, triangular_mesh)