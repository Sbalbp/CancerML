#!/usr/bin/env python
#coding:utf-8

""" This script creates the 5 folds used in the experiments reported in [1].

    It ASSUMES:
        - that the path is relative to the current directory.
        - that BreaKHis_v1.tar.gz was decompressed in current directory.

    It REQUIRES:
        - text files dsfold1.txt, ... dsfold5.txt located in current directory.

    ****Approximately 20 GB of disk space will be allocated for fold1,... fold5 directories.


    -------
    [1] Spanhol, F.A.; Oliveira, L.S.; Petitjean, C.; Heutte, L. "A Dataset for Breast Cancer Histopathological Image Classification". Biomedical Engineering, IEEE Transactions on. Year: 2015, DOI: 10.1109/TBME.2015.2496264

"""
__author__ = "Sergio Balbuena Pantigas"
__email__ = "sbalbp@gmail.com"


import os
import cv2

def rescale_img(image_name, width = 350, height = 230, output_dir = "."):
	name_split = image_name.split('/')[-1].split('.')
	output_file = '%s/%s_rescale_%dx%d.%s' % (output_dir, name_split[0], width, height, name_split[-1])

	img = cv2.imread(image_name)
	img_rescaled = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
	cv2.imwrite(output_file, img_rescaled)

def rescale_dataset(current_dir, origin = './BreaKHis_v1', dest = './BreaKHis_v1_rescaled', width = 350, height = 230):
	if current_dir:
		current_origin = '%s/%s' % (origin, current_dir)
		current_dest = '%s/%s' % (dest, current_dir)
	else:
		current_origin = origin
		current_dest = '%s_%dx%d' % (dest, width, height)

	if not os.path.isdir(current_dest):
		print('CREATING DIRECTORY: %s' % current_dest)
		os.mkdir(current_dest)
	for file_name in os.listdir(current_origin):
		file_path = '%s/%s' % (current_origin, file_name)

		# Navigate recursively to next directory
		if os.path.isdir(file_path):
			print('ENTER DIRECTORY: %s' % file_path)
			rescale_dataset(file_name, current_origin, current_dest, width, height)
		# Rescaled images
		elif file_name.split('.')[-1] in ['png','PNG']:
			print('RESCALING IMAGE: %s' % file_path)
			rescale_img(file_path, width, height, output_dir = current_dest)


"""
# -----------------------------------------------------------------------------
def create_folds_from_ds(dst_path='.', folds=(1,2,3,4,5)):
    #Creates a structure of directories containing images
    #    selected from BreaKHis_v1 dataset.
    
    root_dir = './BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}

    for nfold in folds:
        # directory for nth-fold
        dst_dir = dst_path + '/fold%s' % nfold
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        dst_dir = dst_dir + '/%s'

        # image list
        db = open('dsfold%s.txt' % nfold)

        for row in db.readlines():
            columns = row.split('|')
            imgname = columns[0]
            mag = columns[1]  # 40, 100, 200, or 400
            grp = columns[3].strip()  # train or test

            dst_subdir = dst_dir % grp
            if not os.path.exists(dst_subdir):
                os.mkdir(dst_subdir)

            dst_subdir = dst_subdir + '/%sX/' % mag
            if not os.path.exists(dst_subdir):
                os.mkdir(dst_subdir)

            tumor = imgname.split('-')[0].split('_')[-1]
            srcfile = srcfiles[tumor]

            s = imgname.split('-')
            sub = s[0] + '_' + s[1] + '-' + s[2]

            srcfile = srcfile % (root_dir, sub, mag, imgname)

            dstfile = dst_subdir + imgname

            print "Copying from [%s] to [%s]" % (srcfile, dstfile)
            shutil.copy(srcfile, dstfile)
        print '\n\n\t\tFold #%d finished.\n' % nfold
    db.close()
    print "\nProcess completed."
# -----------------------------------------------------------------------------
"""

if __name__ == "__main__":
	rescale_dataset('')
