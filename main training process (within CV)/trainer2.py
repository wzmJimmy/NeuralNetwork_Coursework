from keras.models import load_model
from keras import backend as K
from keras import Model
from keras import optimizers

import os
import numpy as np
import tensorflow as tf
from Resumable_Trainer import ResumableTrainer_callback
from Serialized_Callback import ReduceLROnPlateau_picklable, EarlyStopping_picklable,ModelCheckpoint_resumable
from lovasz_losses_tf import lovasz_grad,lovasz_hinge,lovasz_hinge_flat,flatten_binary_scores


direct = "output/"
path_pickle = direct + "temp_pickle"
path_best_model = direct + "best.model"
path_temp_model = direct + "temp.model"

best = 0
lr = 1e-3
monitor = 'val_my_iou_metric_2'
max_epochs = 300
epoch_per_turn = 100
patience_es = 25
patience_lr = 20
factor = 0.7
min_lr = 1e-5
tol = 1e-4
verb = 2

'''self-difined function for model'''
## lovasz_loss inside model fitting ##
def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    logits = y_pred  # Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss

## iou inside model fitting ##
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)

'''custom_objects'''
custom_objects1={'my_iou_metric': my_iou_metric}
custom_objects2={'my_iou_metric_2': my_iou_metric_2,'lovasz_loss': lovasz_loss}

'''change model'''
if os.path.isfile(path_best_model):
    model = load_model(path_best_model, custom_objects=custom_objects1)
    input_x = model.layers[0].input
    output_layer = model.layers[-1].input
    model = Model(input_x, output_layer)

    c = optimizers.adam(lr=lr)
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    model.save(path_temp_model, overwrite=True)

'''trainer with callbacks initialization'''
# early_stopping = EarlyStopping_picklable(patience=patience_es, min_delta=tol, monitor=monitor, mode='max', verbose=verb,reset = False)
model_checkpoint = ModelCheckpoint_resumable(path_best_model, monitor=monitor, mode='max', save_best_only=True, verbose=verb)
reduce_lr = ReduceLROnPlateau_picklable(factor=factor, monitor=monitor, mode='max', patience=patience_lr, min_lr=min_lr,verbose=verb,reset = False)
if best!=0:
    model_checkpoint.best = best
    # early_stopping.best = best
    reduce_lr.best = best

mytrainer = ResumableTrainer_callback(max_epochs, epoch_per_turn, path_pickle, custom_objects=custom_objects2,callbacks=[ model_checkpoint, reduce_lr])
if not os.path.isdir(direct): os.mkdir(direct)
mytrainer.save_trainer()
print("trainer initialization finished.")