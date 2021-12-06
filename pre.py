from detr_tf.optimizers import setup_optimizers
import sys
import detr_tf
import tensorflow as tf
from detr_tf.training_config import TrainingConfig
from os.path import expanduser
import os
from detr_tf.data import load_tfcsv_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf import training


class CustomConfig(TrainingConfig):

    def __init__(self):
        super().__init__()
        # Dataset info
        self.data.data_dir = "/Users/wangzhuo/Desktop/detr-tensorflow/Workers"
        # The model is trained using fixed size images.
        # The following is the desired target image size, but it can be change based on your
        # dataset
        self.image_size = (480, 720)
        # Batch size
        self.batch_size = 1
        # Using the target batch size , the training loop will agregate the gradient on 38 steps
        # before to update the weights
        self.target_batch = 8


config = CustomConfig()
train_iterator, class_names = load_tfcsv_dataset(
    config, config.batch_size, True, exclude=["person"], ann_file="train/annotations.csv", img_dir="train")
valid_iterator, class_names = load_tfcsv_dataset(
    config, config.batch_size, False, exclude=["person"], ann_file="test/annotations.csv",img_dir ="test")
print("class_names", class_names)
detr = get_detr_model(config, include_top=False, nb_class=3, weights="detr")
detr.summary()

from detr_tf.networks.detr import get_detr_model
detr = get_detr_model(config,include_top=False,nb_class=3,weights='detr')
detr.summary()

config.train_backbone = tf.Variable(False)
config.train_transformers = tf.Variable(False)
config.train_nlayers = tf.Variable(True)

config.nlayers_lr = tf.Variable(0.001)
# Setup the optimziers and the trainable variables
optimzers = setup_optimizers(detr, config)
training.fit(detr, train_iterator, optimzers, config, epoch_nb=0, class_names=class_names)




