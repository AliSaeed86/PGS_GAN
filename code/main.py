# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
'''
Directory Hierarchy
├── codes
│   ├── dataset.py
│   ├── evaluation.py
│   ├── main.py
│   ├── model.py
│   ├── solver.py
│   ├── TensorFlow_utils.py
│   ├── utils.py
│   ├── DRIVE    here you see the result of training/testing you images, models will be saved in './codes/{}/model_{}_{}_{}'.format(dataset, disriminator, train_interval, batch_size)' folder, e.g., './codes/DRIVE/model_image_100_1' folder.
│   │                                                                    smapled images will be saved in './codes/{}/sample__{}_{}_{}'.format(dataset, disriminator, train_interval, batch_size)', e.g., './codes/DRIVE/sample_image_100_1' folder.
│   ├── STARE    here you see the result of training/testing you images  here you see the result of training/testing you images, models will be saved in './codes/{}/model_{}_{}_{}'.format(dataset, disriminator, train_interval, batch_size)' folder, e.g., './codes/DRIVE/model_image_100_1' folder.
│   │                                                                    smapled images will be saved in './codes/{}/sample__{}_{}_{}'.format(dataset, disriminator, train_interval, batch_size)', e.g., './codes/DRIVE/sample_image_100_1' folder.
├── data         Here you have to put your training/testing images
│   ├── DRIVE
│   └── STARE
├── evaluation  (get after running evaluation.py)
│   ├── DRIVE
│   └── STARE
├── results
│   ├── DRIVE
│   └── STARE

codes: source codes
data: original data. File hierarchy is modified for convenience.
evaluation: quantitative and qualitative evaluation. (get after running evaluation.py)
results: results of other methods. These image files are retrieved from https://cvlsegmentation.github.io/driu/downloads.html.

************ Evaluation **************
Note: Copy predicted vessel images to the ./results/[Drishti-GS1|DRIVE|STARE]/V-GAN folder 
      then bring the segmented images of other methods (that we need to compare with them), and put them in the same folder ./results/[Drishti-GS1|DRIVE|STARE]/method_name
      
Results                are generated in evaluation folder. Hierarchy of the folder is
├── DRIVE
│   ├── comparison     compare difference maps between V-GAN and gold standard
│   ├── measures       measure the AUC_ROC and AUC_PR curves
│   └── vessels        vessels superimposed on segmented masks
└── STARE
    ├── comparison     compare difference maps between V-GAN and gold standard
    ├── measures       measure the AUC_ROC and AUC_PR curves
    └── vessels        vessels superimposed on segmented masks
Area Under the Curve (AUC), Precision and Recall (PR), Receiver Operating Characteristic (ROC     

                                                                                          
The best model is saved based on the sum of the AUC_PR and AUC_ROC on validation data
                                                                                          '''

'''NOTE ADDED BY ALI  : Vessels / optic disc / optic cups/ masks images must be changed from mode--> RGB   to mode--> GrayScale using GIMP app '''

######### Tensorboard viewing 
#''' To view the Tensorboard, open current Environment terminal from the ANACONDA NAVIGATOR and past this line
#inside the terminal:   
#(keras122_tf115__GPU) C:\Users\User> tensorboard --logdir "D:/Spider_IDE/3rd Objective/3    V_GAN    github   (Here)/V-GAN-tensorflow-master/codes/STARE/logs/image_1_1" '''


import os
import tensorflow as tf
from solver import Solver


    

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    # This is to avoid showing every details on the consol while importing cuda from our library 

tf.compat.v1.config.experimental.list_physical_devices('GPU')
#to see all the GPUs
tf.config.experimental.list_physical_devices('GPU')
#to see all the devices
tf.config.experimental.list_physical_devices(device_type=None)
#use this line to check if there is GPU detected. It says True if it detects the available gpu
tf.test.is_gpu_available()

############# This is to prevent tensorflow from allocating all GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##############################################
##############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
##############################################

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#########################################

# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
if physical_devices:
     tf.config.experimental.set_virtual_device_configuration(
         physical_devices[0],
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])




# tf.keras.backend.clear_session()    #should clear the previous model. From https://keras.io/backend/ Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers. After running and saving one model, clear the session, then run next model

####################  These can be executed only once    ##ADDED BY ALI
FLAGS = tf.flags.FLAGS

#### Added by Ali    To delete all the values in the flag if we want to set another values
# for name in list(FLAGS):
#       delattr(FLAGS,name)
################

### Note:  Set higher training intervals between generator and discriminator, which can boost performance a little bit as paper mentioned. However, the mathematical theory behind this experimental results is not clear.
tf.flags.DEFINE_integer('train_interval', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg', 10, 'ratio of gan loss to seg loss, default: 10')
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('discriminator', 'pixel', 'type of discriminator [pixel|patch1|patch2|image], default: patch1')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'RimOneV3_singleGAN', 'dataset name [DRIVE|STARE|Drishti-GS1|RimOneV3], default: Drishti-GS1')    ## here we have to specify the folder name that containg the images of the dataset we need to use i.e., Drishti-GS1/DRIVE/STARE/
tf.flags.DEFINE_bool('is_test', False, 'default: False (train)')               # False for train and True for Test ## Added by Ali
tf.flags.DEFINE_bool('Use_both_GAN', False, 'default: True (to train both GANs)')             

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 11600, 'number of iteratons, default: 50000')    #11600 used in our training
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'models are saved here')
tf.flags.DEFINE_string('sample_dir', './sample', 'sample are saved here')
tf.flags.DEFINE_string('test_dir', './test', 'test images are saved here')
#####################################################################
    
tf.flags.DEFINE_integer('train_interval_OC', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg_OC', 10, 'ratio of gan loss to seg loss, default: 10')
# tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('discriminator_OC', 'pixel', 'type of discriminator [pixel|patch1|patch2|image], default: patch1')
# tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
# tf.flags.DEFINE_string('dataset_OC', 'Drishti-GS1', 'dataset name [DRIVE|STARE], default: Drishti-GS1')    ## here we have to specify the folder name that containg the images of the dataset we need to use i.e., Drishti-GS1/DRIVE/STARE/
# tf.flags.DEFINE_bool('is_test', True, 'default: False (train)')               # False for train and True for Test ## Added by Ali

# tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
# tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
# tf.flags.DEFINE_integer('iters', 50000, 'number of iteratons, default: 50000')
# tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
# tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
# tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')

tf.flags.DEFINE_string('checkpoint_dir_OC', './checkpoints', 'models are saved here')
tf.flags.DEFINE_string('sample_dir_OC', './sample', 'sample are saved here')
tf.flags.DEFINE_string('test_dir_OC', './test', 'test images are saved here')
    
# '''Added by Ali'''    
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    # This is to avoid showing every details on the consol while importing cuda from our library 
# tf.compat.v1.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.list_physical_devices('GPU')             #to see all the GPUs
# tf.config.experimental.list_physical_devices(device_type=None)  #to see all the devices
# tf.test.is_gpu_available()                #use this line to check if there is GPU detected. It says True if it detects the available gpu
# '''########################'''


# '''Added by Ali'''    
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    # This is to avoid showing every details on the consol while importing cuda from our library 
# tf.compat.v1.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.list_physical_devices('GPU')             #to see all the GPUs
# tf.config.experimental.list_physical_devices(device_type=None)  #to see all the devices
# tf.test.is_gpu_available()                #use this line to check if there is GPU detected. It says True if it detects the available gpu
# '''########################'''

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    # tf.compat.v1.reset_default_graph     ## ADDED By Ali to rest the graph for multiple modeling building (multiple execution)  

    solver = Solver(FLAGS)
    if FLAGS.is_test:
        solver.test()
    if not FLAGS.is_test:
        solver.train()


'''
 ملاحظة عند مقارنة النتائج وطلعت اقل من البقية 
 يمكنك ان تعمل المقارنة مع الاطباء حيث ان الاختلافات بينهم كبيرة 
واذا كانت النتائج مالموديل تقع ضمن مدى الاختلافات مالاطباء فيمكن 
اعتبار الموديل نموذجي لانه يقوم بنقس عمل الاطباء 
علما انه لايمكن تصميم موديل 100% وذلك لان تقييمه 
سوف يكون على اساس انوتيشن ممكن ان يكون غلط وبهذه الحالة سوف تعتبر النتيجة 
 الصحيحة للموديل كنتيجة خاطئة ولذلك لايمكن
 ابدا ان يكون لدينا نموذج دقته 100%  
 '''

if __name__ == '__main__':
    tf.app.run()



# t_vars = tf.trainable_variables()
# d_vars = [var for var in t_vars if 'd_OC' in var.name]
# for var in d_vars:
#     print(var.name, var.shape)



''' Notes By Ali:
    patch2 discriminator performs better in Disc rather than image discriminator
    patch2 discriminator performs better in Cup rather than image discriminator
    
    patch2 discriminator performs less in Cup segmentation compared to Disc segmentation (which is very good)
    
    Suggenestions:
        1. we can use patch2 discriminator for Disc segmetation and use patch1 for Cup segmentation (after testing its performacnce on Cup)
        2. for the classifier, we can start training the classifier after a new weights saved for both GANs, in this case we have to build two more generators for OD and OC and load the saved wieths on them during training and fed them with images to train a classified, and these wieght changed if another new model is saved.
'''




'''#####################################################'''
''' To extract the values from the charts stored in tensorboard and move them into excel sheet'''
###### pip install tensorboardX openpyxl
###### pip install tensorflow openpyxl
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# import openpyxl
# ####### Load and parse the events file

# file_path = 'C:/Users/User/Desktop/losses comparison ... OLD/image discriminator/image Adv+L1 + Focal/logs/OC_1_1'
# event_acc = EventAccumulator(file_path)
# event_acc.Reload()
# ####### Check available tags
# tags = event_acc.Tags()
# print(tags)  # Print the ava

# k=0
# list1=['auc_pr_summary_OC', 'auc_roc_summary_OC', 'dice_coeff_summary_OC', 'acc_summary_OC', 'sensitivity_summary_OC', 'specificity_summary_OC', 'score_summary_OC']
# list1=['auc_pr_summary', 'auc_roc_summary', 'dice_coeff_summary', 'acc_summary', 'sensitivity_summary', 'specificity_summary', 'score_summary']

# loss_values = event_acc.scalars.Items(list1[k])
# ##### Create a new workbook and select the active sheet
# workbook = openpyxl.Workbook()
# sheet = workbook.active
# ##### Write the column headers
# sheet['A1'] = 'Step'
# # sheet['B1'] = 'score_summary_OC'
# sheet['B1'] = list1[k]
# ##### Write the loss values to the cells
# for i, value in enumerate(loss_values):
#     step = value.step
#     loss = value.value
#     sheet.cell(row=i+2, column=1).value = step
#     sheet.cell(row=i+2, column=2).value = loss
#     # sheet.cell(row=i+2, column=kk+2).value = loss
# k+=1
# ##### Save the workbook
# workbook.save('auc_pr_summary.xlsx')
'''#############################################################'''







# img = cv2.imread('drishtiGS_035.png')
# plt.imshow(img[:, :, 2], cmap='Greys_r')


# # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.imshow(img2, cmap='Reds_r')


# a = np.zeros((156816, 36, 53806), dtype='uint8')
# a.nbytes


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# #create a 512x512 black image
# img=cv2.imread('drishtiGS_002_DELETE ME.png')
# plt.imshow(img)
# img=np.zeros((512,512,3),np.uint8)
# img.shape

# img2 = img[:,:,0]
# img2= np.asarray(img2)

# dim1= img2.shape[0]
# dim2= img2.shape[1]

# print("\n\n kkkkkkkkkk")
# print(img2.shape)
# print(dim1)
# print(dim2)

# #finding the first top cell with value > 10 
# xx1=0
# for i in range((dim1)):
#     for j in range((dim1)):
#         if img2[i][j] > 0: #10:               ## we put 10 to execlude outliners like 1,2,3,...
#             top_i= i
#             top_j = j
#             xx1=1
#             # print("\n\n AAAAAAAAAAAAAAAAAAAAAAAAA")
#         if xx1 == 1:
#             break

# #finding the last bottom cell with value > 10 
# i,j,xx2=0,0,0   
# for i in reversed(range((dim1))):
#     for j in reversed(range((dim1))):
#         if img2[i][j] > 0: #10:
#             below_i= i
#             below_j = j
#             # print("\n\n BBBBBBBBBBB")
#             xx2=1
#         if xx2 == 1:
#             break    
    
# #finding the first left cell with value > 10 
# i,j,xx3=0,0,0   
# for i in range((dim1)):
#     for j in range((dim1)):
#         if img2[j][i] > 0: #10:
#             left_i= i
#             left_j = j
#             # print("\n\n cCCCCCCCCCCCCCC")
#             xx3=1
#         if xx3 == 1:
#             break
            

# #finding the last right cell with value > 10 
# i,j,xx4=0,0,0   
# dim1= img2.shape[0]
# for i in reversed(range((dim1))):
#     for j in reversed(range((dim1))):
#         if img2[j][i] > 0: #10:
#             right_i= i
#             right_j = j
#             # print("\n\n DDDDDDDDDDDDDD")
#             xx4=1
#         if xx4 == 1:
#             break    
    
# if xx1 == 1 and xx2 == 1 and xx3 == 1 and xx4 == 1:
    
    
#     #finding the vertical diameter and its Middle point
#     Ver_diameter = int((below_i - top_i))
#     Ver_center_point_in_diameter = top_i + int((abs(below_i - top_i))/2)
    
#     #finding the Horizontal diameter and its Middle point
#     Hor_diameter = int((right_i - left_i))
#     Hor_center_point_in_diameter = min(right_i, left_i) + int((abs(right_i - left_i))/2)
    
#     # print("\n\n Ver_diameter")
#     # print(Ver_diameter)
#     # print("\n\n Hor_diameter")
#     # print(Hor_diameter)
#     ## move the found points to x and y for simplicity
#     x=Ver_center_point_in_diameter
#     y=Hor_center_point_in_diameter
    
    
    
#     #non filled circle
#     # img1 = cv2.circle(img,(x,y),(int(Hor_diameter/2)), (0,255,0), 8)
#     # img1.shape
#     # # filled circle
#     # img1 = cv2.circle(img,(256,256),63, (0,0,255), -1)
    
#   # # # img33 = cv2.circle(img,(y,x),(int(Ver_diameter/2)), (255,255,255), -1)  #-1)
#    # ## img33 = cv2.ellipse(img,(y,x),(Ver_diameter,90,(0,0,255), -1)  #-1)
       
#     img33  = cv2.ellipse(img ,(y,x),(int(Hor_diameter/2)-5,int(Ver_diameter/2)),0,0,360,(255,0,255),-1) 
#     ## (y,x) is the center point of ellipse
#     ## (int(Ver_diameter/2),100)  is the redius length of the ellipse in horizontal axis, and the length of the ellipse in vertical axis and we put (-5) to form the ellipse shape not the circle shape 
#     ## ,0,0
#     ## 360  this is to draw the four quarters of the ellipse as 45 is for one quearter only, 90 is for two and 180 and 380 as so on
#     ## (255,0,255)  this is the color of the drawn ellipse
#     ## -1 is to fill the ellipse with color however 0 is to draw borders only
       
       
#     #now use a frame to show it just as displaying a image
#     plt.imshow(img33)
#     img33.shape
#     # cv2.imshow("Circle",img1)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    





# except:
#     # printing stack trace
#     traceback.print_exc()


    # tf.app.run(debug=True, use_reloader=False)
    # tf.app.run(debug=True, use_reloader=False)
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2 
# img = cv2.imread('im0003_DELETE ME.ppm')
# img .shape
# img2 = cv2.imread('drishtiGS_002_DELETE ME.png')
# img2 .shape
# # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.imshow(img2, cmap='Reds_r')


# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# im = imageio.imread('drishtiGS_035.png')
# im_r = np.zeros(np.shape(im))
# im_r[:, :, 0] = im[:, :, 0]
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# axs[0, 0].imshow(im[:, :, 0])
# axs[1, 0].imshow(im[:, :, 0], cmap='Greens_r')   # for gray image 'Greys_r'
# axs[0, 1].imshow(im[:, :, 0], cmap='Reds_r')
# axs[1, 1].imshow(im_r.astype(int))

# axs[0, 0].set_title('pure imshow of 2D-array (R-channel)')
# axs[1, 0].set_title('imshow of 2D-array with cmap="Grey_r"')
# axs[0, 1].set_title('imshow of 2D-array with cmap="Reds_r"')
# axs[1, 1].set_title('imshow of 3D-array with coordinates 1 and 2 \n(i.e.: channels G and B) set to 0')
# plt.tight_layout()










# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import cv2
# img = cv2.imread('drishtiGS_002_DELETE ME.png')    # Notice there is difference between reading with cv2 and reading with mpimg in viewing the image 
# img = mpimg.imread('drishtiGS_002_DELETE ME.png')
# # img=np.asarray(img).astype(np.float32) 
# # b,g,r = cv2.split(img)

# plt.imshow(img)
# Blue_img = img[:,:,0] # Use the Blue channel  from chatGPT
# plt.imshow(Blue_img) ; plt.axis('off')
#  ### link of below code is https://stackoverflow.com/questions/52632718/display-image-only-in-rred-channel-using-python
# greens_r_img = img[:,:,1] # Use the green channel from chatGPT
# plt.imshow(greens_r_img,cmap='Greens_r') ; plt.axis('off')

# red_img = img[:,:,2] # Use the red channel from chatGPT
# plt.imshow(red_img, cmap='Reds_r'); plt.axis('off')
# img22 = img[:,:,2] # Use the red channel from chatGPT
# plt.imshow(img22)
# #b,g,r = cv2.split(img22)
# img222 = np.stack([img22, img22, img22], axis=-1)
# img222.shape

# imgb = np.stack([g, g,g], axis=-1)
# plt.subplot(2,2,2)
# plt.imshow(imgb)

# img = np.stack([r, r, r], axis=-1)
# plt.subplot(2,2,3)
# plt.imshow(img)

# img = np.stack([g, g, g], axis=-1)
# plt.subplot(2,2,4)
# plt.imshow(img)

# # img = np.stack([b, np.zeros_like(b), np.zeros_like(b)], axis=-1)
# plt.imshow(img)
# img.shape



# path= "D:/Spider_IDE/3rd Objective/3. V_GAN  (Here)/V-GAN- (Here is my work)/data/RimOneV3/test/images"

# list_names=[]
# for filename in os.listdir(path):
#     list_names.append(filename)
    
    
    
    
    
    
    
    
    