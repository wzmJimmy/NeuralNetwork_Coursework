import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import resize

from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img

from tqdm import tqdm
import time
from lovasz_losses_tf import lovasz_grad,lovasz_hinge,lovasz_hinge_flat,flatten_binary_scores

path_input= "../input/"
path_sub = "submission.csv"
path_models = ['../' + i +'output/best.model' for i in ('try-2/n3/','a4/','cv/cv1/')]
threshold_bests = [0.131313,0.040404,0]

path_training_input= "training/"
path_test_depth = path_training_input+"depth.csv"

stime = time.time()
img_size_ori = 101
img_size_target = 101 #128
## helper function for resample picture ##
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

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


## IoU metrics with threshold -- copied -- could revise ##
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


## find best threshold ##
# thresholds = np.linspace(0, 1, 100)
# ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])
# threshold_best_index = np.argmax(ious)  # np.argmax(ious[9:-10]) + 9
# iou_best = ious[threshold_best_index]
# threshold_best = thresholds[threshold_best_index]
# np.savetxt(path_thres, np.c_[thresholds, ious])

## submission prepration -- copied ##
# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

## final submission ##
test_df = pd.read_csv(path_test_depth, index_col="id")
x_test = np.array([upsample(np.array(load_img(path_input+"test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
models = [load_model(i,custom_objects=custom_objects2) for i in path_models]
preds_tests = np.array([np.round(model.predict(x_test)>threshold_bests[i]) for i,model in enumerate(models)])
preds_test = np.sum(preds_tests,0)/3

pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]>0.5))) for i, idx in enumerate(tqdm(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(path_sub)
etime = time.time()
print("time consumed:" + str(etime-stime))

