# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils


class CGAN(object):
    def __init__(self, sess, flags, image_size,model_type):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.model_type = model_type

        self.alpha_recip = 1. / self.flags.ratio_gan2seg if self.flags.ratio_gan2seg > 0 else 0
        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c, self.dis_c = 32, 32
        
       
        self._build_net()       
        self._init_assign_op()  

        print('Initialized CGAN SUCCESS!\n')

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, *self.image_size, 3], name='image')
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size, 1], name='Disc')

        if self.model_type == 'Disc':
            self.g_samples = self.generator(self.X,'g_')                     
            self.real_pair = tf.concat([self.X, self.Y], axis=3)
            self.fake_pair = tf.concat([self.X, self.g_samples], axis=3)
    
            d_real, d_logit_real = self.discriminator(self.real_pair, name = 'd_')      
            d_fake, d_logit_fake = self.discriminator(self.fake_pair,name='d_', is_reuse=True)
            
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_real)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
            self.d_loss = self.d_loss_real + self.d_loss_fake
            
            if self.flags.Use_both_GAN == False:
                gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
                self.g_loss = self.alpha_recip * gan_loss 
            
            else:
                gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
                seg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_samples, labels=self.Y))
                l1_loss = tf.reduce_mean(tf.abs(self.g_samples - self.Y))       ### Adding L1 Loss
                # self.g_loss = self.alpha_recip * gan_loss + seg_loss + self.alpha_recip * l1_loss
    
                # Compute the dice loss            
                epsilon = 1e-8                                                             
                intersect = tf.reduce_sum(self.g_samples * self.Y, axis=[1, 2, 3])          
                union = tf.reduce_sum(self.g_samples, axis=[1, 2, 3]) + tf.reduce_sum(self.Y, axis=[1, 2, 3])   
                dice_loss = tf.reduce_mean((2. * intersect + epsilon) / (union + epsilon))  
                # # self.g_loss = self.alpha_recip * gan_loss + seg_loss + self.alpha_recip * l1_loss + self.alpha_recip * dice_loss
    
                ## Compute the focal loss
                alpha=0.25
                gamma=2
                bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.g_samples)
                ## Compute the modulating factor for the focal loss
                pt = tf.exp(-bce_loss)
                modulating_factor = (1 - pt) ** gamma
                focal_loss = alpha * bce_loss * modulating_factor
    
                self.g_loss = self.alpha_recip * gan_loss + seg_loss + self.alpha_recip * l1_loss + self.alpha_recip  * focal_loss   + self.alpha_recip * dice_loss
                #######################
        
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]
    
            dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
                .minimize(self.d_loss, var_list=d_vars)
            dis_ops = [dis_op] + self._dis_train_ops
            self.dis_optim = tf.group(*dis_ops)
    
            gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
                .minimize(self.g_loss, var_list=g_vars)
            gen_ops = [gen_op] + self._gen_train_ops
            self.gen_optim = tf.group(*gen_ops)
            
        else: 
            self.g_samples_OC = self.generator(self.X,'g_OC')                    
            self.real_pair_OC = tf.concat([self.X, self.Y], axis=3)
            self.fake_pair_OC = tf.concat([self.X, self.g_samples_OC], axis=3)
    
            d_real_OC, d_logit_real_OC = self.discriminator(self.real_pair_OC, name = 'd_OC')  
            d_fake_OC, d_logit_fake_OC = self.discriminator(self.fake_pair_OC, name = 'd_OC', is_reuse=True)
        

            self.d_loss_real_OC = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real_OC, labels=tf.ones_like(d_real_OC)))
            self.d_loss_fake_OC = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake_OC, labels=tf.zeros_like(d_logit_fake_OC)))
            self.d_loss_OC = self.d_loss_real_OC + self.d_loss_fake_OC
        
            gan_loss_OC = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake_OC, labels=tf.ones_like(d_logit_fake_OC)))
            seg_loss_OC = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_samples_OC, labels=self.Y))
            l1_loss_OC = tf.reduce_mean(tf.abs(self.g_samples_OC - self.Y))     
             
            ## Compute the dice loss
            epsilon_OC = 1e-8                                                           
            intersect_OC = tf.reduce_sum(self.g_samples_OC * self.Y, axis=[1, 2, 3])          
            union_OC = tf.reduce_sum(self.g_samples_OC, axis=[1, 2, 3]) + tf.reduce_sum(self.Y, axis=[1, 2, 3])  
            dice_loss_OC = tf.reduce_mean((2. * intersect_OC + epsilon_OC) / (union_OC + epsilon_OC))  
      
            ## Compute the focal loss
            alpha=0.25
            gamma=2
            bce_loss_OC = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.g_samples_OC)
            ## Compute the modulating factor for the focal loss
            pt_OC = tf.exp(-bce_loss_OC)
            modulating_factor_OC = (1 - pt_OC) ** gamma
            focal_loss_OC = alpha * bce_loss_OC * modulating_factor_OC

            self.g_loss_OC = self.alpha_recip * gan_loss_OC + seg_loss_OC + self.alpha_recip * l1_loss_OC +self.alpha_recip  * focal_loss_OC  + self.alpha_recip * dice_loss_OC

           
           
            t_vars_OC = tf.trainable_variables()
            d_vars_OC = [var for var in t_vars_OC if 'd_OC' in var.name]
            g_vars_OC = [var for var in t_vars_OC if 'g_OC' in var.name]
    
            dis_op_OC = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
                .minimize(self.d_loss_OC, var_list=d_vars_OC)
            dis_ops_OC = [dis_op_OC] + self._dis_train_ops
            self.dis_optim_OC = tf.group(*dis_ops_OC)
    
            gen_op_OC = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
                .minimize(self.g_loss_OC, var_list=g_vars_OC)
            gen_ops_OC = [gen_op_OC] + self._gen_train_ops
            self.gen_optim_OC = tf.group(*gen_ops_OC)

             

    def _init_assign_op(self):
        if self.model_type == 'Disc':
            self.best_auc_sum_placeholder = tf.placeholder(tf.float32, name='best_auc_sum_placeholder')
            self.auc_pr_placeholder = tf.placeholder(tf.float32, name='auc_pr_placeholder')
            self.auc_roc_placeholder = tf.placeholder(tf.float32, name='auc_roc_placeholder')
            self.dice_coeff_placeholder = tf.placeholder(tf.float32, name='dice_coeff_placeholder')
            self.acc_placeholder = tf.placeholder(tf.float32, name='acc_placeholder')
            self.sensitivity_placeholder = tf.placeholder(tf.float32, name='sensitivity_placeholder')
            self.specificity_placeholder = tf.placeholder(tf.float32, name='specificity_placeholder')
            self.IOU_disc_placeholder = tf.placeholder(tf.float32, name='IOU_disc_placeholder')
            self.score_placeholder = tf.placeholder(tf.float32, name='score_placeholder')
    
            auc_pr = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_pr')
            auc_roc = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_roc')
            dice_coeff = tf.Variable(0., trainable=False, dtype=tf.float32, name='dice_coeff')
            acc = tf.Variable(0., trainable=False, dtype=tf.float32, name='acc')
            sensitivity = tf.Variable(0., trainable=False, dtype=tf.float32, name='sensitivity')
            specificity = tf.Variable(0., trainable=False, dtype=tf.float32, name='specificity')
            IOU_disc = tf.Variable(0., trainable=False, dtype=tf.float32, name='IOU_disc')
            self.best_auc_sum = tf.Variable(0., trainable=False, dtype=tf.float32, name='best_auc_sum')
            score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score')
    
            self.best_auc_sum_assign_op = self.best_auc_sum.assign(self.best_auc_sum_placeholder)
            auc_pr_assign_op = auc_pr.assign(self.auc_pr_placeholder)
            auc_roc_assign_op = auc_roc.assign(self.auc_roc_placeholder)
            dice_coeff_assign_op = dice_coeff.assign(self.dice_coeff_placeholder)
            acc_assign_op = acc.assign(self.acc_placeholder)
            sensitivity_assign_op = sensitivity.assign(self.sensitivity_placeholder)
            specificity_assign_op = specificity.assign(self.specificity_placeholder)
            IOU_disc_assign_op = IOU_disc.assign(self.IOU_disc_placeholder)
            score_assign_op = score.assign(self.score_placeholder)
    
            self.measure_assign_op = tf.group(auc_pr_assign_op, auc_roc_assign_op, dice_coeff_assign_op,
                                              acc_assign_op, sensitivity_assign_op, specificity_assign_op,
                                              IOU_disc_assign_op,
                                              score_assign_op)
            
            
            # for tensorboard
            if not self.flags.is_test:
                self.writer = tf.summary.FileWriter("{}/train/{}/logs/{}_{}".format(
                    self.flags.dataset, self.flags.discriminator, self.flags.train_interval, self.flags.batch_size))
    
            auc_pr_summ = tf.summary.scalar("auc_pr_summary", auc_pr)
            auc_roc_summ = tf.summary.scalar("auc_roc_summary", auc_roc)
            dice_coeff_summ = tf.summary.scalar("dice_coeff_summary", dice_coeff)
            acc_summ = tf.summary.scalar("acc_summary", acc)
            sensitivity_summ = tf.summary.scalar("sensitivity_summary", sensitivity)
            specificity_summ = tf.summary.scalar("specificity_summary", specificity)
            IOU_disc_summ = tf.summary.scalar("IOU_summary", IOU_disc)
            score_summ = tf.summary.scalar("score_summary", score)
    
            self.measure_summary = tf.summary.merge([auc_pr_summ, auc_roc_summ, dice_coeff_summ, acc_summ,
                                                     sensitivity_summ, specificity_summ,
                                                     # IOU_disc_summ,
                                                     score_summ])

        else:    
    
            self.best_auc_sum_placeholder_OC = tf.placeholder(tf.float32, name='best_auc_sum_placeholder_OC')
            self.auc_pr_placeholder_OC = tf.placeholder(tf.float32, name='auc_pr_placeholder_OC')
            self.auc_roc_placeholder_OC = tf.placeholder(tf.float32, name='auc_roc_placeholder_OC')
            self.dice_coeff_placeholder_OC = tf.placeholder(tf.float32, name='dice_coeff_placeholder_OC')
            self.acc_placeholder_OC = tf.placeholder(tf.float32, name='acc_placeholder_OC')
            self.sensitivity_placeholder_OC = tf.placeholder(tf.float32, name='sensitivity_placeholder_OC')
            self.specificity_placeholder_OC = tf.placeholder(tf.float32, name='specificity_placeholder_OC')
            self.IOU_cup_placeholder = tf.placeholder(tf.float32, name='IOU_cup_placeholder')
            self.score_placeholder_OC = tf.placeholder(tf.float32, name='score_placeholder_OC')
    
            self.best_auc_sum_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='best_auc_sum_OC')
            auc_pr_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_pr_OC')
            auc_roc_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='auc_roc_OC')
            dice_coeff_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='dice_coeff_OC')
            acc_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='acc_OC')
            sensitivity_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='sensitivity_OC')
            specificity_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='specificity_OC')
            IOU_cup = tf.Variable(0., trainable=False, dtype=tf.float32, name='IOU_cup')
            score_OC = tf.Variable(0., trainable=False, dtype=tf.float32, name='score_OC')
    
            self.best_auc_sum_assign_op_OC = self.best_auc_sum_OC.assign(self.best_auc_sum_placeholder_OC)
            auc_pr_assign_op_OC = auc_pr_OC.assign(self.auc_pr_placeholder_OC)
            auc_roc_assign_op_OC = auc_roc_OC.assign(self.auc_roc_placeholder_OC)
            dice_coeff_assign_op_OC = dice_coeff_OC.assign(self.dice_coeff_placeholder_OC)
            acc_assign_op_OC = acc_OC.assign(self.acc_placeholder_OC)
            sensitivity_assign_op_OC = sensitivity_OC.assign(self.sensitivity_placeholder_OC)
            specificity_assign_op_OC = specificity_OC.assign(self.specificity_placeholder_OC)
            IOU_cup_assign_op = IOU_cup.assign(self.IOU_cup_placeholder)
            score_assign_op_OC = score_OC.assign(self.score_placeholder_OC)
    
            self.measure_assign_op_OC = tf.group(auc_pr_assign_op_OC, auc_roc_assign_op_OC, dice_coeff_assign_op_OC,
                                              acc_assign_op_OC, sensitivity_assign_op_OC, specificity_assign_op_OC,
                                               IOU_cup_assign_op,
                                              score_assign_op_OC)
            
    
            if not self.flags.is_test:
                # self.writer = tf.summary.FileWriter("{}/train/logs_OC/{}_{}_{}".format(
                self.writer = tf.summary.FileWriter("{}/train/{}/logs/OC_{}_{}".format(    
                    self.flags.dataset, self.flags.discriminator_OC, self.flags.train_interval_OC, self.flags.batch_size))

    
            auc_pr_summ = tf.summary.scalar("auc_pr_summary_OC", auc_pr_OC)
            auc_roc_summ = tf.summary.scalar("auc_roc_summary_OC", auc_roc_OC)
            dice_coeff_summ = tf.summary.scalar("dice_coeff_summary_OC", dice_coeff_OC)
            acc_summ = tf.summary.scalar("acc_summary_OC", acc_OC)
            sensitivity_summ = tf.summary.scalar("sensitivity_summary_OC", sensitivity_OC)
            specificity_summ = tf.summary.scalar("specificity_summary_OC", specificity_OC)
            IOU_cup_summ = tf.summary.scalar("IOU_cup_summary", IOU_cup)
            score_summ = tf.summary.scalar("score_summary_OC", score_OC)
    
            self.measure_summary_OC = tf.summary.merge([auc_pr_summ, auc_roc_summ, dice_coeff_summ, acc_summ,
                                                     sensitivity_summ, specificity_summ,
                                                      IOU_cup_summ,
                                                     score_summ])
            
    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            # conv1: (N, 640, 640, 1) -> (N, 320, 320, 32)
            conv1 = tf_utils.conv2d(data, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._gen_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._gen_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 320, 320, 32) -> (N, 160, 160, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._gen_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2-batch2', _ops=self._gen_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 160, 160, 64) -> (N, 80, 80, 128)
            conv3 = tf_utils.conv2d(pool2, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._gen_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._gen_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            # conv4: (N, 80, 80, 128) -> (N, 40, 40, 256)
            conv4 = tf_utils.conv2d(pool3, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._gen_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._gen_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            # conv5: (N, 40, 40, 256) -> (N, 40, 40, 512)
            conv5 = tf_utils.conv2d(pool4, 16*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._gen_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._gen_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')

            # conv6: (N, 40, 40, 512) -> (N, 80, 80, 256)
            up1 = tf_utils.upsampling2d(conv5, size=(2, 2), name='conv6_up')
            conv6 = tf.concat([up1, conv4], axis=3, name='conv6_concat')
            conv6 = tf_utils.conv2d(conv6, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv1')
            conv6 = tf_utils.batch_norm(conv6, name='conv6_batch1', _ops=self._gen_train_ops)
            conv6 = tf.nn.relu(conv6, name='conv6_relu1')
            conv6 = tf_utils.conv2d(conv6, 8*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv2')
            conv6 = tf_utils.batch_norm(conv6, name='conv6_batch2', _ops=self._gen_train_ops)
            conv6 = tf.nn.relu(conv6, name='conv6_relu2')

            # conv7: (N, 80, 80, 256) -> (N, 160, 160, 128)
            up2 = tf_utils.upsampling2d(conv6, size=(2, 2), name='conv7_up')
            conv7 = tf.concat([up2, conv3], axis=3, name='conv7_concat')
            conv7 = tf_utils.conv2d(conv7, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv1')
            conv7 = tf_utils.batch_norm(conv7, name='conv7_batch1', _ops=self._gen_train_ops)
            conv7 = tf.nn.relu(conv7, name='conv7_relu1')
            conv7 = tf_utils.conv2d(conv7, 4*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv2')
            conv7 = tf_utils.batch_norm(conv7, name='conv7_batch2', _ops=self._gen_train_ops)
            conv7 = tf.nn.relu(conv7, name='conv7_relu2')

            # conv8: (N, 160, 160, 128) -> (N, 320, 320, 64)
            up3 = tf_utils.upsampling2d(conv7, size=(2, 2), name='conv8_up')
            conv8 = tf.concat([up3, conv2], axis=3, name='conv8_concat')
            conv8 = tf_utils.conv2d(conv8, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv1')
            conv8 = tf_utils.batch_norm(conv8, name='conv8_batch1', _ops=self._gen_train_ops)
            conv8 = tf.nn.relu(conv8, name='conv8_relu1')
            conv8 = tf_utils.conv2d(conv8, 2*self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv2')
            conv8 = tf_utils.batch_norm(conv8, name='conv8_batch2', _ops=self._gen_train_ops)
            conv8 = tf.nn.relu(conv8, name='conv8_relu2')

            # conv9: (N, 320, 320, 64) -> (N, 640, 640, 32)
            up4 = tf_utils.upsampling2d(conv8, size=(2, 2), name='conv9_up')
            conv9 = tf.concat([up4, conv1], axis=3, name='conv9_concat')
            conv9 = tf_utils.conv2d(conv9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv1')
            conv9 = tf_utils.batch_norm(conv9, name='conv9_batch1', _ops=self._gen_train_ops)
            conv9 = tf.nn.relu(conv9, name='conv9_relu1')
            conv9 = tf_utils.conv2d(conv9, self.gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv2')
            conv9 = tf_utils.batch_norm(conv9, name='conv9_batch2', _ops=self._gen_train_ops)
            conv9 = tf.nn.relu(conv9, name='conv9_relu2')

            # output layer: (N, 640, 640, 32) -> (N, 640, 640, 1)
            output = tf_utils.conv2d(conv9, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output)

    def discriminator(self, data,name, is_reuse=False):
        if self.flags.discriminator == 'pixel':
            return self.discriminator_pixel(data,name, is_reuse=is_reuse)
        elif self.flags.discriminator == 'patch1':
            return self.discriminator_patch1(data,name, is_reuse=is_reuse)
        elif self.flags.discriminator == 'patch2':
            return self.discriminator_patch2(data,name, is_reuse=is_reuse)
        elif self.flags.discriminator == 'image':
            return self.discriminator_image(data,name, is_reuse=is_reuse)
        else:
            raise NotImplementedError


    def discriminator_pixel(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 640, 640, 4) -> (N,, 640, 640, 32)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu1')

            # conv2: (N, 640, 640, 32) -> (N, 640, 640, 64)
            conv2 = tf_utils.conv2d(conv1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.lrelu(conv2)

            # conv3: (N, 640, 640, 64) -> (N, 640, 640, 128)
            conv3 = tf_utils.conv2d(conv2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.lrelu(conv3)

            # output layer: (N, 640, 640, 128) -> (N, 640, 640, 1)
            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_patch2(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 640, 640, 4) -> (N,, 160, 160, 32)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 160, 160, 32) -> (N, 80, 80, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 80, 80, 64) -> (N, 80, 80, 128)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')

            # output layer: (N, 80, 80, 128) -> (N, 80, 80, 1)
            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_patch1(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 640, 640, 4) -> (N,, 160, 160, 32)
            print(' ---------- print(data) ----------\n')
            print(data)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')
            print(' ---------- print(Pool1) ----------\n')
            print(pool1)

            # conv2: (N, 160, 160, 32) -> (N, 40, 40, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')
            print(' ---------- print(Pool2) ----------\n')
            print(pool2)
            # conv3: (N, 40, 40, 64) -> (N, 20, 20, 128)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')
            print(' ---------- print(Pool3) ----------\n')
            print(pool3)
            # conv4: (N, 20, 20, 128) -> (N, 10, 10, 256)
            conv4 = tf_utils.conv2d(pool3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')
            print(' ---------- print(Pool4) ----------\n')
            print(pool4)
            # conv5: (N, 10, 10, 256) -> (N, 10, 10, 512)
            conv5 = tf_utils.conv2d(pool4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')
            print(' ---------- print(conv5) ----------\n')
            print(conv5)
            # output layer: (N, 10, 10, 512) -> (N, 10, 10, 1)
            output = tf_utils.conv2d(conv5, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')
            
            print(' ---------- print(output) ----------\n')
            print(output)
            
            return tf.nn.sigmoid(output), output

    def discriminator_image(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # conv1: (N, 640, 640, 4) -> (N,, 160, 160, 32)
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=self._dis_train_ops)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 160, 160, 32) -> (N, 40, 40, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf_utils.batch_norm(conv2, name='conv2_batch2', _ops=self._dis_train_ops)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            # conv3: (N, 40, 40, 64) -> (N, 20, 20, 128)
            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=self._dis_train_ops)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            # conv4: (N, 20, 20, 128) -> (N, 10, 10, 256)
            conv4 = tf_utils.conv2d(pool3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=self._dis_train_ops)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            # conv5: (N, 10, 10, 256) -> (N, 10, 10, 512)
            conv5 = tf_utils.conv2d(pool4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=self._dis_train_ops)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')

            # output layer: (N, 10, 10, 512) -> (N, 1, 1, 512) -> (N, 1)
            shape = conv5.get_shape().as_list()
            gap = tf.layers.average_pooling2d(inputs=conv5, pool_size=shape[1], strides=1, padding='VALID',
                                              name='global_vaerage_pool')
            gap_flatten = tf.reshape(gap, [-1, 16*self.dis_c])
            output = tf_utils.linear(gap_flatten, 1, name='linear_output')

            return tf.nn.sigmoid(output), output

    def train_dis(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run discriminator
        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed_dict)
        return d_loss


    def train_dis_OC(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run discriminator
        _, d_loss_OC = self.sess.run([self.dis_optim_OC, self.d_loss_OC], feed_dict=feed_dict)
        return d_loss_OC

    def train_gen(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run generator
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed_dict)
        return g_loss
    

    def train_gen_OC(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        # run generator
        _, g_loss_OC = self.sess.run([self.gen_optim_OC, self.g_loss_OC], feed_dict=feed_dict)
        return g_loss_OC



    def measure_assign(self, auc_pr, auc_roc, dice_coeff, acc, sensitivity, specificity, IOU, score, iter_time,model_type):
        if model_type == "Disc":
            feed_dict = {self.auc_pr_placeholder: auc_pr,
                         self.auc_roc_placeholder: auc_roc,
                         self.dice_coeff_placeholder: dice_coeff,
                         self.acc_placeholder: acc,
                         self.sensitivity_placeholder: sensitivity,
                         self.specificity_placeholder: specificity,
                          self.IOU_disc_placeholder: IOU,
                         self.score_placeholder: score}
    
            self.sess.run(self.measure_assign_op, feed_dict=feed_dict)
    
            summary = self.sess.run(self.measure_summary)
            self.writer.add_summary(summary, iter_time)

        else:
            feed_dict = {self.auc_pr_placeholder_OC: auc_pr,
                         self.auc_roc_placeholder_OC: auc_roc,
                         self.dice_coeff_placeholder_OC: dice_coeff,
                         self.acc_placeholder_OC: acc,
                         self.sensitivity_placeholder_OC: sensitivity,
                         self.specificity_placeholder_OC: specificity,
                          self.IOU_cup_placeholder: IOU,
                         self.score_placeholder_OC: score}
    
            self.sess.run(self.measure_assign_op_OC, feed_dict=feed_dict)
    
            summary = self.sess.run(self.measure_summary_OC)
            self.writer.add_summary(summary, iter_time)
            
            
    def best_auc_sum_assign(self, auc_sum):
        self.sess.run(self.best_auc_sum_assign_op, feed_dict={self.best_auc_sum_placeholder: auc_sum})


    def best_auc_sum_assign_OC(self, auc_sum):
        self.sess.run(self.best_auc_sum_assign_op_OC, feed_dict={self.best_auc_sum_placeholder_OC: auc_sum})

        
    def sample_imgs(self, x_data):
        return self.sess.run(self.g_samples, feed_dict={self.X: x_data})


    def sample_imgs_OC(self, x_data):
        return self.sess.run(self.g_samples_OC, feed_dict={self.X: x_data})
 
    
    
