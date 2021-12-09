from detr_tf.optimizers import setup_optimizers
import sys
import detr_tf
import tensorflow as tf
from detr_tf.training_config import TrainingConfig
from os.path import expanduser
import os
from detr_tf.data import load_tfcsv_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf import training
from detr_tf.inference import get_model_inference, numpy_bbox_to_image
import matplotlib.pyplot as plt
import numpy as np
import eval


class CustomConfig(TrainingConfig):

    def __init__(self):
        super().__init__()
        # Dataset info45
        self.data.data_dir = "Workers"
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

# detr.load_weights('/Users/yuxuanzhao/detr-tensorflow/model/project')
detr.summary()

config.train_backbone = tf.Variable(False)
config.train_transformers = tf.Variable(False)
config.train_nlayers = tf.Variable(True)

config.nlayers_lr = tf.Variable(0.001)

optimzers = setup_optimizers(detr, config)

training.fit(detr, train_iterator, optimzers, config, epoch_nb=0, class_names=class_names)

count = 5
for valid_images, target_bbox, target_class in valid_iterator:
    if count == 0:
        break
    count -= 1
    m_outputs = detr(valid_images, training=False)
    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, config.background_class,
                                                                bbox_format="xy_center")

    result = numpy_bbox_to_image(
        np.array(valid_images[0]),
        np.array(predicted_bbox),
        np.array(predicted_labels),
        scores=np.array(predicted_scores),
        class_name=class_names,
        config=config
    )
    plt.imshow(result)
    plt.show()

eval.eval_model(detr, config, class_names, valid_iterator)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

config.train_transformers.assign(True)
config.transformers_lr.assign(1e-4)
config.nlayers_lr.assign(1e-3)

for epoch in range(1, 4):
    training.fit(detr, train_iterator, optimzers, config, epoch_nb=epoch, class_names=class_names)

    count = 5
    for valid_images, target_bbox, target_class in valid_iterator:
        if count == 0:
            break
        count -= 1
        m_outputs = detr(valid_images, training=False)
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, config.background_class,
                                                                                 bbox_format="xy_center")

        result = numpy_bbox_to_image(
            np.array(valid_images[0]),
            np.array(predicted_bbox),
            np.array(predicted_labels),
            scores=np.array(predicted_scores),
            class_name=class_names,
            config=config
        )
        plt.imshow(result)
        plt.show()

eval.eval_model(detr, config, class_names, valid_iterator)
