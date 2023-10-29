# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Jaemin Son
# ---------------------------------------------------------
import os
import random
import numpy as np
from datetime import datetime

import utils as utils


class Dataset(object):
    def __init__(self, dataset, flags):
        self.dataset = dataset
        self.flags = flags

        ######## ADDED BY ALI
        # if self.dataset == 'DRIVE':
        #     self.image_size = (640, 640) 
        # elif self.dataset == 'STARE':
        #     self.image_size = (720, 720)
            
        # else:  #  then it is (Drishti-GS1)
        #     self.image_size = (640, 640)
            
        self.image_size = (640, 640) if self.dataset == 'DRIVE' else  (720, 720)
        self.ori_shape = (584, 565) if self.dataset == 'DRIVE' else  (605, 700)       # Any dataset other than (DRIVE) using in this code, we have to set its images to 700x605
        self.val_ratio = 0.1  # 10% of the training data are used as validation data
        self.train_dir = "D:/Spider_IDE/3rd Objective/3. V_GAN  (Here)/V-GAN- (Here is my work)/data/{}/training/".format(self.dataset)          # specifying training images directory
        self.test_dir = "D:/Spider_IDE/3rd Objective/3. V_GAN  (Here)/V-GAN- (Here is my work)/data/{}/test/".format(self.dataset)               # specifying testing images directory

        self.num_train, self.num_val, self.num_test = 0, 0, 0

        self._read_data()  # read training, validation, and test data from the function (_read_data(self):)
        print('num of training images: {}'.format(self.num_train))
        print('num of validation images: {}'.format(self.num_val))
        print('num of test images: {}'.format(self.num_test))

    def _read_data(self):
        #'''#############      here is to get the images for TESTing'''
        if self.flags.is_test:
            # real test images and Discs in the memory
            self.test_imgs, self.test_Discs, self.test_masks, self.test_mean_std, self.test_Cups = utils.get_test_imgs(
                target_dir=self.test_dir, img_size=self.image_size, dataset=self.dataset)
            self.test_img_files = utils.all_files_under(os.path.join(self.test_dir, 'images'))

            self.num_test = self.test_imgs.shape[0]

       # '''#############      here is to get the images for Training'''
        elif not self.flags.is_test:
            random.seed(datetime.now())  # set random seed
            self.train_img_files, self.train_Disc_files, mask_files,self.Cup_files = utils.get_img_path(
                self.train_dir, self.dataset)

            ################# This is to set the size of training and validation sets
            self.num_train = int(len(self.train_img_files))
            self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
            self.num_train -= self.num_val
            ###########################
            
            ################# moving validation images/Discs/Masks/Cups to their variables to be used later
            self.val_img_files = self.train_img_files[-self.num_val:]
            self.val_Disc_files = self.train_Disc_files[-self.num_val:]
            val_mask_files = mask_files[-self.num_val:]
            self.val_Cup_files = self.Cup_files[-self.num_val:]
            ################################
            
            ############## to remove the images/Discs that are moved to the validation set, remove them from training set
            self.train_img_files = self.train_img_files[:-self.num_val]
            self.train_Disc_files = self.train_Disc_files[:-self.num_val]
            self.Cup_files = self.Cup_files[:-self.num_val]
            ##############################
            
            # read val images and Discs in the memory
            self.val_imgs, self.val_Discs, self.val_masks, self.val_mean_std, self.val_Cups = utils.get_val_imgs(
                self.val_img_files, self.val_Disc_files, val_mask_files, self.val_Cup_files, img_size=self.image_size)

            self.num_val = self.val_imgs.shape[0]

    def train_next_batch(self, batch_size, model_type):
        train_indices = np.random.choice(self.num_train, batch_size, replace=True)
        # if model_type == "Disc":
        #     train_imgs, train_Discs, train_Cups = utils.get_train_batch(self.train_img_files, self.train_Disc_files, self.Cup_files,
        #                                                             train_indices.astype(np.int32), img_size=self.image_size, model_type= "Disc")
        # else:
        #     train_imgs, train_Discs, train_Cups = utils.get_train_batch(self.train_img_files, self.train_Disc_files, self.Cup_files,
        #                                                             train_indices.astype(np.int32), img_size=self.image_size, model_type= "Cup")

        train_imgs, train_Discs, train_Cups = utils.get_train_batch(self.train_img_files, self.train_Disc_files, self.Cup_files,
                                                                train_indices.astype(np.int32), img_size=self.image_size, model_type= "Disc")
        train_Discs = np.expand_dims(train_Discs, axis=3)
        train_Cups = np.expand_dims(train_Cups, axis=3)
        
        return train_imgs, train_Discs, train_Cups 
