import os
import numpy as np
import tensorflow as tf
from Resumable_Trainer import ResumableTrainer_callback
from Serialized_Callback import ReduceLROnPlateau_picklable, EarlyStopping_picklable,ModelCheckpoint_resumable

direct = "output/"
path_pickle = direct + "temp_pickle"
path_best_model = direct + "best.model"
path_temp_model = direct + "temp.model"

best = 0
lr = 1e-2
monitor = 'val_my_iou_metric'
max_epochs = 300
epoch_per_turn = 300
# patience_es = 20
patience_lr = 10
factor = 0.5
min_lr = 1e-5
tol = 1e-5
verb = 2

'''self-difined function for model'''
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

'''custom_objects'''
custom_objects1={'my_iou_metric': my_iou_metric}

'''trainer with callbacks initialization'''
model_checkpoint = ModelCheckpoint_resumable(path_best_model, monitor=monitor, mode='max', save_best_only=True, verbose=verb)
reduce_lr = ReduceLROnPlateau_picklable(factor=factor, monitor=monitor, mode='max', patience=patience_lr, min_lr=min_lr,verbose=verb, reset=False)

mytrainer = ResumableTrainer_callback(max_epochs, epoch_per_turn, path_pickle,custom_objects=custom_objects1, callbacks=[model_checkpoint, reduce_lr])

if not os.path.isdir(direct): os.mkdir(direct)
mytrainer.save_trainer()
print("trainer initialization finished.")
