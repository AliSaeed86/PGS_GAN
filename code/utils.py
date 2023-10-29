# ---------------------------------------------------------
# Tensorflow Disc-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Jaemin Son
# ---------------------------------------------------------
import os
import sys
import math

import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance
from skimage import filters
from scipy.ndimage import rotate
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix



def get_img_path(target_dir, dataset):
    img_files, Disc_files, mask_files = None, None, None
    if dataset == 'DRIVE':
        img_files, Disc_files, mask_files = DRIVE_files(target_dir)
    elif dataset == 'STARE':
        img_files, Disc_files, mask_files,Cup_files = STARE_files(target_dir)
    else: 
        img_files, Disc_files, mask_files,Cup_files = OTHER_DB_files_for_OD(target_dir)
    return img_files, Disc_files, mask_files,Cup_files 


def OTHER_DB_files_for_OD(data_path):
    print('\n\n\n data_path is ' + str(data_path))
    img_dir = os.path.join(data_path, "images")
    print(img_dir)
    Disc_dir = os.path.join(data_path, "1st_manual") 
    print(Disc_dir)
    mask_dir = os.path.join(data_path, "mask")
    print(mask_dir)
    Cup_dir = os.path.join(data_path, "Cropped_OC")
    print(Cup_dir)
    print('\n\n\n')
    img_files = all_files_under(img_dir, extension=".jpg") 
    Disc_files = all_files_under(Disc_dir, extension=".jpg")
    mask_files = all_files_under(mask_dir, extension=".jpg")
    Cup_files = all_files_under(Cup_dir, extension=".jpg")

    return img_files, Disc_files, mask_files,Cup_files

def STARE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    Disc_dir = os.path.join(data_path, "1st_manual")
    mask_dir = os.path.join(data_path, "mask")
    Cup_dir = os.path.join(data_path, "Cropped_OC")
    
    img_files = all_files_under(img_dir, extension=".png")
    Disc_files = all_files_under(Disc_dir, extension=".png")
    mask_files = all_files_under(mask_dir, extension=".png")
    Cup_files = all_files_under(Cup_dir, extension=".png")
    
    return img_files, Disc_files, mask_files,Cup_files


# noinspection PyPep8Naming
def DRIVE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    Disc_dir = os.path.join(data_path, "1st_manual")
    mask_dir = os.path.join(data_path, "mask")

    img_files = all_files_under(img_dir, extension=".tif")
    Disc_files = all_files_under(Disc_dir, extension=".gif")
    mask_files = all_files_under(mask_dir, extension=".gif")

    return img_files, Disc_files, mask_files


def load_images_under_dir(path_dir):
    files = all_files_under(path_dir)
    return imagefiles2arrs(files)


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

import matplotlib.pyplot as plt
import numpy as np
import cv2 

def imagefiles2arrs(filenames, model_type="Disc"):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    if len(img_shape) == 3:
        if model_type == "Disc":
            for file_index in range(len(filenames)):
                img22 = cv2.imread(filenames[file_index])
                img22 = img22[:,:,2] # Use the red channel from chatGPT
                img = np.stack([img22, img22, img22], axis=-1)
                images_arr[file_index] = np.asarray(img).astype(np.float32)  
    else:
        for file_index in range(len(filenames)):
            img = Image.open(filenames[file_index])
            images_arr[file_index] = np.asarray(img).astype(np.float32)
    return images_arr


def get_train_batch(train_img_files, train_Disc_files, train_Cup_files, train_indices, img_size, model_type):
    batch_size = len(train_indices)
    batch_img_files, batch_Disc_files, batch_Cup_files  = [], [], []
    for _, idx in enumerate(train_indices):
        batch_img_files.append(train_img_files[idx])
        batch_Disc_files.append(train_Disc_files[idx])
        batch_Cup_files.append(train_Cup_files[idx])

    # load images
    fundus_imgs = imagefiles2arrs(batch_img_files) 
    Disc_imgs = imagefiles2arrs(batch_Disc_files) / 255
    Cup_imgs = imagefiles2arrs(batch_Cup_files) / 255
    
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    Disc_imgs = pad_imgs(Disc_imgs, img_size)
    Cup_imgs = pad_imgs(Cup_imgs, img_size)
    
    assert (np.min(Disc_imgs) == 0 and np.max(Disc_imgs) == 1)
    assert (np.min(Cup_imgs) == 0 and np.max(Cup_imgs) == 1)

    # random mirror flipping
    for idx in range(batch_size):
        if np.random.random() > 0.5:
            fundus_imgs[idx] = fundus_imgs[idx, :, ::-1, :] 
            Disc_imgs[idx] = Disc_imgs[idx, :, ::-1]  
            Cup_imgs[idx] = Cup_imgs[idx, :, ::-1]  

    # random rotation
    for idx in range(batch_size):
        angle = np.random.randint(360)
        fundus_imgs[idx] = random_perturbation(rotate(input=fundus_imgs[idx], angle=angle, axes=(0, 1),
                                                      reshape=False, order=1))
        Disc_imgs[idx] = rotate(input=Disc_imgs[idx], angle=angle, axes=(0, 1), reshape=False, order=1)
        Cup_imgs[idx] = rotate(input=Cup_imgs[idx], angle=angle, axes=(0, 1), reshape=False, order=1)

    # z score with mean, std of each image
    for idx in range(batch_size):
        mean = np.mean(fundus_imgs[idx, ...][fundus_imgs[idx, ..., 0] > 40.0], axis=0)    
        std = np.std(fundus_imgs[idx, ...][fundus_imgs[idx, ..., 0] > 40.0], axis=0)      

        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[idx, ...] = (fundus_imgs[idx, ...] - mean) / std
        
    return fundus_imgs, np.round(Disc_imgs), np.round(Cup_imgs)


def get_val_imgs(img_files, Disc_files, mask_files, Cup_files, img_size):
    # load images
    fundus_imgs = imagefiles2arrs(img_files)
    Disc_imgs = imagefiles2arrs(Disc_files) / 255
    mask_imgs = imagefiles2arrs(mask_files) / 255
    Cup_imgs = imagefiles2arrs(Cup_files) / 255
    
    # padding
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    Disc_imgs = pad_imgs(Disc_imgs, img_size)
    mask_imgs = pad_imgs(mask_imgs, img_size)
    Cup_imgs = pad_imgs(Cup_imgs, img_size)

    assert (np.min(Disc_imgs) == 0 and np.max(Disc_imgs) == 1)
    assert (np.min(mask_imgs) == 0 and np.max(mask_imgs) == 1)
    assert (np.min(Cup_imgs) == 0 and np.max(Cup_imgs) == 1)

    # augmentation
    all_fundus_imgs = [fundus_imgs]
    all_Disc_imgs = [Disc_imgs]
    all_mask_imgs = [mask_imgs]
    all_Cup_imgs = [Cup_imgs]

    flipped_imgs = fundus_imgs[:, :, ::-1, :]  
    flipped_Discs = Disc_imgs[:, :, ::-1]
    flipped_masks = mask_imgs[:, :, ::-1]
    flipped_Cups = Cup_imgs[:, :, ::-1]

    all_fundus_imgs.append(flipped_imgs)
    all_Disc_imgs.append(flipped_Discs)
    all_mask_imgs.append(flipped_masks)
    all_Cup_imgs.append(flipped_Cups)

    for angle in range(3, 360, 3): 
        print("Val data augmentation {} degree...".format(angle))
        all_fundus_imgs.append(random_perturbation(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False,
                                                          order=1)))
        all_fundus_imgs.append(random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False,
                                                          order=1)))
        
        all_Disc_imgs.append(rotate(Disc_imgs, angle, axes=(1, 2), reshape=False, order=1))
        all_Disc_imgs.append(rotate(flipped_Discs, angle, axes=(1, 2), reshape=False, order=1))

        all_mask_imgs.append(rotate(mask_imgs, angle, axes=(1, 2), reshape=False, order=1))
        all_mask_imgs.append(rotate(flipped_masks, angle, axes=(1, 2), reshape=False, order=1))
        
        all_Cup_imgs.append(rotate(Cup_imgs, angle, axes=(1, 2), reshape=False, order=1))
        all_Cup_imgs.append(rotate(flipped_Cups, angle, axes=(1, 2), reshape=False, order=1))

    fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
    Disc_imgs = np.concatenate(all_Disc_imgs, axis=0)
    mask_imgs = np.concatenate(all_mask_imgs, axis=0)
    Cup_imgs = np.concatenate(all_Cup_imgs, axis=0)
    mean_std = []
    
    n_all_imgs = fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean = np.mean(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)    
        std = np.std(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)      
        
        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[index, ...] = (fundus_imgs[index, ...] - mean) / std

        mean_std.append({'mean': mean, 'std': std})

    return fundus_imgs, np.round(Disc_imgs), np.round(mask_imgs), mean_std, np.round(Cup_imgs)


def get_test_imgs(target_dir, img_size, dataset):
    img_files, Disc_files, mask_files, mask_imgs = None, None, None, None
    if dataset == 'DRIVE':
        img_files, Disc_files, mask_files = DRIVE_files(target_dir)
    elif dataset == 'STARE':
        img_files, Disc_files, mask_files = STARE_files(target_dir)
    else: 
        img_files, Disc_files, mask_files,Cup_files = OTHER_DB_files_for_OD(target_dir)

    # load images
    fundus_imgs = imagefiles2arrs(img_files)
    Disc_imgs = imagefiles2arrs(Disc_files) / 255
    Cup_imgs = imagefiles2arrs(Cup_files) / 255
    
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    Disc_imgs = pad_imgs(Disc_imgs, img_size)
    Cup_imgs = pad_imgs(Cup_imgs, img_size)
    
    assert (np.min(Disc_imgs) == 0 and np.max(Disc_imgs) == 1)
    assert (np.min(Cup_imgs) == 0 and np.max(Cup_imgs) == 1)

    mask_imgs = imagefiles2arrs(mask_files) / 255
    mask_imgs = pad_imgs(mask_imgs, img_size)
    assert (np.min(mask_imgs) == 0 and np.max(mask_imgs) == 1)

    # z score with mean, std of each image
    mean_std = []
    n_all_imgs = fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean = np.mean(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)
        std = np.std(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)

        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[index, ...] = (fundus_imgs[index, ...] - mean) / std

        mean_std.append({'mean': mean, 'std': std})

    return fundus_imgs, np.round(Disc_imgs), mask_imgs, mean_std , np.round(Cup_imgs)


def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def pad_imgs(imgs, img_size):
    padded = None
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    target_h, target_w = img_size[0], img_size[1]
    if len(imgs.shape) == 4:
        d = imgs.shape[3]
        padded = np.zeros((imgs.shape[0], target_h, target_w, d))
    elif len(imgs.shape) == 3:
        padded = np.zeros((imgs.shape[0], img_size[0], img_size[1]))

    start_h, start_w = (target_h - img_h) // 2, (target_w - img_w) // 2
    end_h, end_w = start_h + img_h, start_w + img_w
    padded[:, start_h:end_h, start_w:end_w, ...] = imgs

    return padded


def crop_to_original(imgs, ori_shape):
    # imgs: (N, 640, 640, 3 or None)
    # ori_shape: (584, 565)
    pred_shape = imgs.shape
    assert len(pred_shape) > 2

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 3:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[:, start_h:end_h, start_w:end_w, :]
        else:  
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[:, start_h:end_h, start_w:end_w]


def random_perturbation(imgs):
    for i in range(imgs.shape[0]):
        im = Image.fromarray(imgs[i, ...].astype(np.uint8))
        en = ImageEnhance.Color(im)
        im = en.enhance(np.random.uniform(0.8, 1.2))
        imgs[i, ...] = np.asarray(im).astype(np.float32)

    return imgs


def pixel_values_in_mask(true_Discs, pred_Discs, masks, split_by_img=False):
    assert np.max(pred_Discs) <= 1.0 and np.min(pred_Discs) >= 0.0
    assert np.max(true_Discs) == 1.0 and np.min(true_Discs) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_Discs.shape[0] == true_Discs.shape[0] and masks.shape[0] == true_Discs.shape[0]
    assert pred_Discs.shape[1] == true_Discs.shape[1] and masks.shape[1] == true_Discs.shape[1]

    if split_by_img:
        n = pred_Discs.shape[0]
        return (np.array([true_Discs[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]),
                np.array([pred_Discs[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]))
    else:
        return true_Discs[masks == 1].flatten(), pred_Discs[masks == 1].flatten()


def remain_in_mask(imgs, masks):
    imgs[masks == 0] = 0
    return imgs


# noinspection PyPep8Naming
def AUC_ROC(true_Disc_arr, pred_Disc_arr):
    """
    Area under the ROC curve with x axis flipped
    ROC: Receiver operating characteristic
    """

    AUC_ROC_ = roc_auc_score(true_Disc_arr.flatten(), pred_Disc_arr.flatten())
    return AUC_ROC_


# noinspection PyPep8Naming
def AUC_PR(true_Disc_arr, pred_Disc_arr):
    """
    Precision-recall curve: sklearn function
    auc: Area Under Curve, sklearn function
    """
    precision, recall, _ = precision_recall_curve(true_Disc_arr.flatten(),
                                                  pred_Disc_arr.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec


def threshold_by_f1(true_Discs, generated, masks, flatten=True, f1_score=False):
    Discs_in_mask, generated_in_mask = pixel_values_in_mask(true_Discs, generated, masks)
    precision, recall, thresholds = precision_recall_curve(
        Discs_in_mask.flatten(), generated_in_mask.flatten(), pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_Discs_bin = np.zeros(generated.shape)
    pred_Discs_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_Discs_bin[masks == 1].flatten(), best_f1
        else:
            return pred_Discs_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_Discs_bin, best_f1
        else:
            return pred_Discs_bin


def best_f1_threshold(precision, recall, thresholds):
    best_f1, best_threshold = -1., None
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold


def threshold_by_otsu(pred_Discs, masks, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_Discs[masks == 1])
    pred_Discs_bin = np.zeros(pred_Discs.shape)
    pred_Discs_bin[pred_Discs >= threshold] = 1

    if flatten:
        return pred_Discs_bin[masks == 1].flatten()
    else:
        return pred_Discs_bin


def dice_coefficient_in_train(true_Disc_arr, pred_Disc_arr):
    true_Disc_arr = true_Disc_arr.astype(np.bool)
    pred_Disc_arr = pred_Disc_arr.astype(np.bool)

    intersection = np.count_nonzero(true_Disc_arr & pred_Disc_arr)

    size1 = np.count_nonzero(true_Disc_arr)
    size2 = np.count_nonzero(pred_Disc_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def misc_measures(true_Disc_arr, pred_Disc_arr):
    cm = confusion_matrix(true_Disc_arr, pred_Disc_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return acc, sensitivity, specificity


def difference_map(ori_Disc, pred_Disc, mask):
    # ori_Disc : an RGB image
    thresholded_Disc = threshold_by_f1(np.expand_dims(ori_Disc, axis=0),
                                         np.expand_dims(pred_Disc, axis=0),
                                         np.expand_dims(mask, axis=0), flatten=False)

    thresholded_Disc = np.squeeze(thresholded_Disc, axis=0)
    diff_map = np.zeros((ori_Disc.shape[0], ori_Disc.shape[1], 3))

    # Green (overlapping)
    diff_map[(ori_Disc == 1) & (thresholded_Disc == 1)] = (0, 255, 0)
    # Red (false negative, missing in pred)
    diff_map[(ori_Disc == 1) & (thresholded_Disc != 1)] = (255, 0, 0)
    # Blue (false positive)
    diff_map[(ori_Disc != 1) & (thresholded_Disc == 1)] = (0, 0, 255)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_Disc == 1) & (thresholded_Disc == 1)])
    fn = len(diff_map[(ori_Disc == 1) & (thresholded_Disc != 1)])
    fp = len(diff_map[(ori_Disc != 1) & (thresholded_Disc == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)


def operating_pts_human_experts(gt_Discs, pred_Discs, masks):
    gt_Discs_in_mask, pred_Discs_in_mask = pixel_values_in_mask(
        gt_Discs, pred_Discs, masks, split_by_img=True)

    n = gt_Discs_in_mask.shape[0]
    op_pts_roc, op_pts_pr = [], []
    for i in range(n):
        cm = confusion_matrix(gt_Discs_in_mask[i], pred_Discs_in_mask[i])
        fpr = 1 - 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        tpr = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        prec = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = tpr
        op_pts_roc.append((fpr, tpr))
        op_pts_pr.append((recall, prec))

    return op_pts_roc, op_pts_pr


def misc_measures_evaluation(true_Discs, pred_Discs, masks):
    thresholded_Disc_arr, f1_score = threshold_by_f1(true_Discs, pred_Discs, masks, f1_score=True)
    true_Disc_arr = true_Discs[masks == 1].flatten()

    cm = confusion_matrix(true_Disc_arr, thresholded_Disc_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return f1_score, acc, sensitivity, specificity


def dice_coefficient(true_Discs, pred_Discs, masks):
    thresholded_Discs = threshold_by_f1(true_Discs, pred_Discs, masks, flatten=False)

    true_Discs = true_Discs.astype(np.bool)
    thresholded_Discs = thresholded_Discs.astype(np.bool)

    intersection = np.count_nonzero(true_Discs & thresholded_Discs)

    size1 = np.count_nonzero(true_Discs)
    size2 = np.count_nonzero(thresholded_Discs)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def save_obj(true_Disc_arr, pred_Disc_arr, auc_roc_file_name, auc_pr_file_name, model_type):
    if model_type == "Disc":
        fpr, tpr, _ = roc_curve(true_Disc_arr, pred_Disc_arr)  # roc_curve: sklearn function
        precision, recall, _ = precision_recall_curve(true_Disc_arr.flatten(),
                                                      pred_Disc_arr.flatten(), pos_label=1)
        with open(auc_roc_file_name, 'wb') as f:
            pickle.dump({"fpr": fpr, "tpr": tpr}, f, pickle.HIGHEST_PROTOCOL)
        with open(auc_pr_file_name, 'wb') as f:
            pickle.dump({"precision": precision, "recall": recall}, f, pickle.HIGHEST_PROTOCOL)

    else:
        fpr, tpr, _ = roc_curve(true_Disc_arr, pred_Disc_arr)  # roc_curve: sklearn function
        precision, recall, _ = precision_recall_curve(true_Disc_arr.flatten(),
                                                      pred_Disc_arr.flatten(), pos_label=1)
        with open(auc_roc_file_name, 'wb') as f:
            pickle.dump({"fpr_OC": fpr, "tpr_OC": tpr}, f, pickle.HIGHEST_PROTOCOL)
        with open(auc_pr_file_name, 'wb') as f:
            pickle.dump({"precision_OC": precision, "recall_OC": recall}, f, pickle.HIGHEST_PROTOCOL)


def print_metrics(itr, kargs):
    print("*** Metrics in Iteration {} For Disc ====> ".format(itr))
    for name, value in kargs.items(): 
        print("{:}, ".format(value))   
    sys.stdout.flush()
  

def print_metrics_OC(itr, kargs):
    print("*** Metrics in Iteration {} For OC ====> ".format(itr))
    for name, value in kargs.items():
       
        print("{:}, ".format(value))   
    sys.stdout.flush()
    print('===================================================')

# noinspection PyPep8Naming
def plot_AUC_ROC(fprs, tprs, method_names, fig_dir, op_pts):
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k'] if len(fprs) == 8 \
        else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 0] if len(fprs) == 8 else [4, 1, 2, 3, 0]

    print("****** ROC AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array "
          "(check <home>/pretrained/auc_roc*.npy)")

    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:.4}".format(method_names[index], auc(fprs[index], tprs[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label='Human')
        else:
            plt.step(fprs[index], tprs[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "ROC.png"))
    plt.close()


# noinspection PyPep8Naming
def plot_AUC_PR(precisions, recalls, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k'] if len(precisions) == 8 \
        else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 0] if len(precisions) == 8 else [4, 1, 2, 3, 0]

    # print auc
    print("****** Precision Recall AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array "
          "(check <home>/pretrained/auc_pr*.npy)")

    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:.4}".format(method_names[index], auc(recalls[index], precisions[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(recalls[index], precisions[index], colors[index] + '*',
                     label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(recalls[index], precisions[index], colors[index] + '*', label='Human')
        else:
            plt.step(recalls[index], precisions[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "Precision_recall.png"))
    plt.close()

def mean_std_precision(path):
    data = np.load(path+'auc_pr.npy', allow_pickle=True)
    data2 = np.load(path+'auc_roc.npy', allow_pickle=True)
   
    data.update(data2) 
    precision = data['precision']
    recall = data['recall']
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    accuracy = (precision + recall) / 2
    specificity = 1 - (data['fpr'] / (data['fpr'] + data['tpr']))
    fpr = data['fpr'] / (data['fpr'] + data['tpr'])
    f1_score = 2 * (precision * recall) / (precision + recall)
    mean_precision = np.mean(precision)
    std_precision = np.std(precision)
    mean_recall = np.mean(recall)
    std_recall = np.std(recall)
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_fpr = np.mean(fpr)
    std_fpr = np.std(fpr)
    mean_f1_score = np.mean(f1_score)
    std_f1_score = np.std(f1_score)
    
    # Print the calculated metrics
    print("\nMean Precision_OC:                                ", mean_precision)
    print("Standard Deviation of Precision_OC:                 ", std_precision)
    print("Mean Recall_OC:                                     ", mean_recall)
    print("Standard Deviation of Recall_OC:                    ", std_recall)
    print("Mean Accuracy_OC:                                   ", mean_accuracy)
    print("Standard Deviation of Accuracy_OC:                  ", std_accuracy)
    print("Mean Specificity_OC:                                ", mean_specificity)
    print("Standard Deviation of Specificity_OC:               ", std_specificity)
    print("Mean False Positive Rate for Cup:                   ", mean_fpr)
    print("Standard Deviation of False Positive Rate for Cup:  ", std_fpr)
    print("Mean F1-score_OC:                                   ", mean_f1_score)
    print("Standard Deviation of F1-score for Cup:             ", std_f1_score)
    print('\n')

def mean_std_precision_OC(path):
    data = np.load(path+'auc_pr_OC.npy', allow_pickle=True)
    data2 = np.load(path+'auc_roc_OC.npy', allow_pickle=True)
    data.update(data2) 
    precision = data['precision_OC']
    recall = data['recall_OC']
    
    plt.plot(recall, precision)
    plt.xlabel('Recall_OC')
    plt.ylabel('Precision_OC')
    plt.title('Precision-Recall Curve_OC')
    plt.show()
    accuracy = (precision + recall) / 2
    specificity = 1 - (data['fpr_OC'] / (data['fpr_OC'] + data['tpr_OC']))

    fpr = data['fpr_OC'] / (data['fpr_OC'] + data['tpr_OC'])
    f1_score = 2 * (precision * recall) / (precision + recall)
    mean_precision = np.mean(precision)
    std_precision = np.std(precision)
    
    mean_recall = np.mean(recall)
    std_recall = np.std(recall)
    
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    
    mean_fpr = np.mean(fpr)
    std_fpr = np.std(fpr)
    
    mean_f1_score = np.mean(f1_score)
    std_f1_score = np.std(f1_score)
    
    # Print the calculated metrics
    print("\nMean Precision_OC:                                ", mean_precision)
    print("Standard Deviation of Precision_OC:                 ", std_precision)
    print("Mean Recall_OC:                                     ", mean_recall)
    print("Standard Deviation of Recall_OC:                    ", std_recall)
    print("Mean Accuracy_OC:                                   ", mean_accuracy)
    print("Standard Deviation of Accuracy_OC:                  ", std_accuracy)
    print("Mean Specificity_OC:                                ", mean_specificity)
    print("Standard Deviation of Specificity_OC:               ", std_specificity)
    print("Mean False Positive Rate for Cup:                   ", mean_fpr)
    print("Standard Deviation of False Positive Rate for Cup:  ", std_fpr)
    print("Mean F1-score_OC:                                   ", mean_f1_score)
    print("Standard Deviation of F1-score for Cup:             ", std_f1_score)
    print('\n')

