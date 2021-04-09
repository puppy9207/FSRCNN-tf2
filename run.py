from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from FSRCNN import FSRCNN
import tensorflow as tf
import data_util
import time
import os
import cv2 
import numpy as np

class run:
    def __init__(self,scale, batch, epochs, params,validdir,checkpoint_dir="./ckpt"):
        self.now = None
        self.scale = scale
        self.batch = batch
        self.epochs = epochs
        self.params = params
        self.validdir = validdir
        self.loss = MeanSquaredError()
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(0.0001),
                                              model=FSRCNN(scale,*params))
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                        directory=checkpoint_dir,
                        max_to_keep=1)
        self.restore()
        
    def psnr(self,x1, x2):
        return tf.image.psnr(x1, x2, max_val=255)

    def resolve(self,model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    def evaluate(self, model, dataset):
        psnr_values = []
        for lr, hr in dataset:
            sr = self.resolve(model, lr)
            psnr_value = self.psnr(hr, sr)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
            
    @property
    def model(self):
        return self.checkpoint.model

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)
        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss_value


    def train(self,imgFolder,steps=300000):
        image_paths = data_util.getpaths(imgFolder)
        train_dataset = tf.data.Dataset.from_generator(generator=data_util.make_dataset, 
                                                 output_types=(tf.float32, tf.float32), 
                                                 output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
                                                 args=[image_paths, self.scale])
        train_dataset = train_dataset.padded_batch(self.batch, padded_shapes=([None, None, 1],[None, None, 1]))
        
        # Create validation dataset
        val_image_paths = data_util.getpaths(self.validdir)
        val_dataset = tf.data.Dataset.from_generator(generator=data_util.make_val_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
                                                 args=[val_image_paths, self.scale])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, 1],[None, None, 1]))

        loss_mean = Mean()
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()
        for i in range(self.epochs):
            for lr, hr in train_dataset.take(steps-ckpt.step.numpy()):
                ckpt.step.assign_add(1)
                step = ckpt.step.numpy()

                loss= self.train_step(lr,hr)
                loss_mean(loss)
                    
                if step % 1000 == 0:
                    loss_value = loss_mean.result()
                    loss_mean.reset_states()

                    psnr_value = self.evaluate(self.checkpoint.model,val_dataset)

                    duration = time.perf_counter() - self.now
                    print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')
                    self.checkpoint.model.save('my_model.h5')
                    if psnr_value <= ckpt.psnr:
                        self.now = time.perf_counter()
                        # skip saving checkpoint, no PSNR improvement
                        continue
                    ckpt.psnr = psnr_value
                    ckpt_mgr.save()

                    self.now = time.perf_counter()
            
    def test(self):
        img = cv2.imread('./data/train/0004.png', 3)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_y = img_ycc[:,:,0]
        floatimg = img_y.astype(np.float32) / 255.0
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)
        model = tf.keras.models.load_model("my_model.h5")
        sr = model(LR_input_)
        Y = sr[0]
        Y = tf.clip_by_value((Y)*255,0, 255)
        Y = tf.cast(Y, tf.uint8)
        Cr = np.expand_dims(cv2.resize(img_ycc[:,:,1], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
        Cb = np.expand_dims(cv2.resize(img_ycc[:,:,2], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
        HR_image = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))
        bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("SR_001.png",HR_image)
        cv2.imwrite("bc_001.png",bicubic_image)
