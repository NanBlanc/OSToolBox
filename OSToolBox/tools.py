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
        list_file.sort(key=natural_keys)
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
        list_file.sort(key=natural_keys)
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
        list_dir.sort(key=natural_keys)
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
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
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
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

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


def write_ply(filename, field_list, field_names, triangular_faces=None):
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

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    
    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print(n_fields,field_names)
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
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
