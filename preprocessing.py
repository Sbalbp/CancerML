from __future__ import division
import os
import math
import argparse
import itertools

import cv2
import numpy as np

VERBOSE = False

if VERBOSE:
    def verboseprint(*args):
        for arg in args:
           print(arg)
else:
    verboseprint = lambda *a: None


def rescale_img(image_name, width = 350, height = 230, output_dir = ".", **kwargs):
    name_split = image_name.split('/')[-1].split('.')
    output_file = '%s/%s_rescale_%dx%d.%s' % (output_dir, name_split[0], width, height, name_split[-1])

    img = cv2.imread(image_name)
    img_rescaled = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    cv2.imwrite(output_file, img_rescaled)

def split_grid_patches(img, width, height):
    maxr = math.floor(img.shape[0] / height)
    maxc = math.floor(img.shape[1] / width)

    coord = list(itertools.product(np.arange(maxr)*height, np.arange(maxc)*width))

    rows = np.array([int(coord[i][0]) for i in range(len(coord))])
    cols = np.array([int(coord[i][1]) for i in range(len(coord))])

    return (rows, cols)

def split_window_patches(img, width, height):
    maxr = math.floor(img.shape[0] / (height/2)) - 1
    maxc = math.floor(img.shape[1] / (width/2)) - 1

    coord = list(itertools.product(np.arange(maxr)*(height/2), np.arange(maxc)*(width/2)))

    rows = np.array([int(coord[i][0]) for i in range(len(coord))])
    cols = np.array([int(coord[i][1]) for i in range(len(coord))])

    return (rows, cols)

def split_random_patches(img, width, height, quantity):
    maxr = img.shape[0] - height
    maxc = img.shape[1] - width

    maxn = maxr * maxc

    values = np.arange(maxn)
    np.random.shuffle(values)
    values = values[:quantity]

    rows = values % maxr
    cols = values // maxr

    return (rows, cols)

def create_random_patches_file(image_name, width = 32, height = 32, quantity = 100, output_dir = ".", **kwargs):
    name_split = image_name.split('/')[-1].split('.')

    img = cv2.imread(image_name)

    maxr = img.shape[0] - height + 1
    maxc = img.shape[1] - width + 1

    rows = np.random.randint(0, maxr, quantity)
    cols = np.random.randint(0, maxc, quantity)

    for patch in range(quantity):
        output_file = '%s/%s_random_patch_%d_%dx%d_%05d.%s' % (output_dir, name_split[0], quantity, width, height, patch, name_split[-1])
        cv2.imwrite(output_file, img[rows[patch]:rows[patch]+height,cols[patch]:cols[patch]+width])

def create_slidingw_patches(image_name, width = 32, height = 32, output_dir = ".", **kwargs):
    name_split = image_name.split('/')[-1].split('.')

    img = cv2.imread(image_name)

    rows = (img.shape[0] // (height // 2)) - 1
    cols = (img.shape[1] // (width // 2)) - 1
    channels = img.shape[2]
    patches = np.lib.stride_tricks.as_strided(img, (rows,cols,height,width,channels), ( (height // 2)*(channels*img.shape[1]), (width // 2)*channels, (channels*img.shape[1]), channels, img.itemsize))

    for r in range(rows):
        for c in range(cols):
            output_file = '%s/%s_sliding_window_patch_%dx%d_%05d.%s' % (output_dir, name_split[0], width, height, r*cols+c, name_split[-1])
            cv2.imwrite(output_file, patches[r][c])

def create_grid_patches(image_name, width = 32, height = 32, output_dir = ".", **kwargs):
    name_split = image_name.split('/')[-1].split('.')

    img = cv2.imread(image_name)

    rows = img.shape[0] // height
    cols = img.shape[1] // width
    channels = img.shape[2]
    patches = np.lib.stride_tricks.as_strided(img, (rows,cols,height,width,channels), (height*(channels*img.shape[1]), width*channels, (channels*img.shape[1]), channels, img.itemsize))

    for r in range(rows):
        for c in range(cols):
            output_file = '%s/%s_grid_patch_%dx%d_%05d.%s' % (output_dir, name_split[0], width, height, r*cols+c, name_split[-1])
            cv2.imwrite(output_file, patches[r][c])

funcs = {'rescale': rescale_img, 'patch_random': create_random_patches_file, 'patch_window': create_slidingw_patches, 'grid': create_grid_patches}

def apply_to_dataset(current_dir, origin = './BreaKHis_v1_rescaled_350x230', dest = './BreaKHis_v1_rescaled_350x230_patches', fapply = None, width = 32, height = 32, **kwargs):
    if not fapply:
        raise Exception('Function to apply not specified')
    elif not fapply in funcs.keys():
        raise Exception('Function \'%s\' is not supported' % fapply)
    else:
        function_to_apply = funcs[fapply]

    if current_dir:
        current_origin = '%s/%s' % (origin, current_dir)
        current_dest = '%s/%s' % (dest, current_dir)
    else:
        current_origin = origin
        current_dest = '%s_%dx%d' % (dest, width, height)

    if not os.path.isdir(current_dest):
        verboseprint('CREATING DIRECTORY: %s' % current_dest)
        os.mkdir(current_dest)
    for file_name in os.listdir(current_origin):
        file_path = '%s/%s' % (current_origin, file_name)

        # Navigate recursively to next directory
        if os.path.isdir(file_path):
            verboseprint('ENTER DIRECTORY: %s' % file_path)
            apply_to_dataset(file_name, current_origin, current_dest, fapply, width, height, **kwargs)
        # Rescaled images
        elif file_name.split('.')[-1] in ['png','PNG']:
            verboseprint('CREATING PATCHES FROM IMAGE: %s' % file_path)
            function_to_apply(file_path, width = width, height = height, output_dir = current_dest, **kwargs)

def calculate_average_img(source, dest, route_filters, width = None, height = None):
    from functools import reduce

    avg_img = np.zeros(0)
    total_img = 0

    for path, subdir, files in os.walk(source):
        if reduce(lambda x, y: x and y, [rfilter in path for rfilter in route_filters]):
            verboseprint('ENTER DIRECTORY: %s' % path)
            for file in files:
                if file.split('.')[-1] in ['png', 'PNG']:
                    #print('ADDING IMAGE: %s' % file)
                    img = cv2.imread('%s/%s' % (path, file))

                    if avg_img.shape[0] == 0:
                        if width == None:
                            width = img.shape[1]
                        if height == None:
                            height = img.shape[0]

                        avg_img = np.zeros((height,width,3))

                    avg_img += img
                    total_img += 1

    avg_img = avg_img / total_img
    if not os.path.isdir(dest):
        verboseprint('CREATING DIRECTORY: %s' % dest)
        os.makedirs(dest)
    cv2.imwrite('%s/average_%s.png' % (dest, reduce(lambda x, y: '%s_%s' % (x,y), route_filters)), avg_img)
    verboseprint('Total images processed: %d' % total_img)

def subtract_average_img(source, dest, route_filters, averages):
    from functools import reduce

    verboseprint('%s/average_%s.png' % (averages, reduce(lambda x, y: '%s_%s' % (x,y), route_filters)))
    avg_img = cv2.imread('%s/average_%s.png' % (averages, reduce(lambda x, y: '%s_%s' % (x,y), route_filters)))

    total_img = 0
    for path, subdir, files in os.walk(source):
        if reduce(lambda x, y: x and y, [rfilter in path for rfilter in route_filters]):
            verboseprint('ENTER DIRECTORY: %s' % path)
            for file in files:
                if file.split('.')[-1] in ['png', 'PNG']:

                    img = cv2.imread('%s/%s' % (path, file))

                    new_img = img - avg_img

                    cv2.imwrite('%s/%s/%s' % (dest, path, file), new_img)
                    total_img += 1

    verboseprint('Total images processed: %d' % total_img)

if __name__ == "__main__":
    # Command line parameters
    parser = argparse.ArgumentParser( description = 'Preprocessing utilities for the image datasets' )
    parser.add_argument( '-s', dest = 'source', metavar = 'Source directory', required = True, help = 'Location of the dataset files' )
    parser.add_argument( '-p', dest = 'preprocessing', metavar = 'Preprocessing', required = True, choices = funcs.keys(), help = 'Preprocessing to apply' )
    parser.add_argument( '-c', dest = 'width', metavar = 'Width', type = int, required = True, help = 'Width of output images' )
    parser.add_argument( '-r', dest = 'height', metavar = 'Height', type = int, required = True, help = 'Height of output images' )
    parser.add_argument( '-q', dest = 'quantity', metavar = 'Quantity', type = int, default = 1000, help = 'Number of random_patches to create (default: 1000)' )
    #parser.add_argument( '--bag', dest = 'use_bagging', action = 'store_true', help = 'Use bagging of weaker networks' )

    args = parser.parse_args()

    source = args.source
    preprocessing = args.preprocessing
    width = args.width
    height = args.height
    quantity = args.quantity

    apply_to_dataset('', source, '%s_%s' % (source, preprocessing), preprocessing, width, height, quantity=quantity)

    """
    processes = []
    for resolution in ['40X','100X','200X','400X']:
        processes.append(Process(target = calculate_average_img, args = (source,'./averages/%s' % source, [resolution],)))

    for process in processes:
        process.start()
    for process in processes:
        process.join()
    """


    #for resolution in ['40X','100X','200X','400X']:
        #calculate_average_img(source, './averages/%s' % source, [resolution])

    """
    for fold in ['fold1','fold2','fold3','fold4','fold5']:
        for trtst in ['train','test']:
            for resolution in ['40X','100X','200X','400X']:
                subtract_average_img(source, '%s_averaged' % source, [fold,trtst,resolution], '/home/sbalbuena/cancer/datasets/averages/folds_rescale_350x230')
    """

