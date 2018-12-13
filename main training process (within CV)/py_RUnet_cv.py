import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

from keras import backend as K
from keras.preprocessing.image import load_img
from keras import Model
from keras import optimizers
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from tqdm import tqdm
import time
import os 
import pickle
from lovasz_losses_tf import lovasz_grad,lovasz_hinge,lovasz_hinge_flat,flatten_binary_scores


direct = "output/"
path_input= "../../input/"
path_training_input = "../training/"

path_pickle = direct + "temp_pickle"
path_history = direct + "history.csv"
path_sub = direct + "submission.csv"
path_thres = direct + "threshold.txt"
path_temp_model = direct + "temp.model"
path_best_model = direct + "best.model"

path_test_depth = path_training_input+"depth.csv"
files = [path_training_input+i+'.npy' for i in ['X','Y', 'trains','valids']]

if not os.path.isfile(path_temp_model):
    seed = 1214
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

batch_size = 32
cv_total = 5
cv_ind = 0
img_size_ori = 101
img_size_target = 101 #128

''' ######### common operations ############# '''
stime = time.time()

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

## data augmentation using imgaug ##
def do_augmentation(seqs, seq2_train, X_train, y_train):
    # Use seq_det to build augmentation.
    seq_det = seqs.to_deterministic()
    X_train_aug = seq2_train.augment_image(seq_det.augment_image(X_train))
    y_train_aug = seq_det.augment_image(y_train)

    if y_train_aug.shape != (101, 101):
        X_train_aug = ia.imresize_single_image(X_train_aug, (101, 101), interpolation="linear")
        y_train_aug = ia.imresize_single_image(y_train_aug, (101, 101), interpolation="nearest")
    return np.array(X_train_aug), np.array(y_train_aug)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip
    iaa.OneOf([
        iaa.Noop(),
        iaa.Affine(rotate=(-6,6), translate_percent={"x": (-0.25,0.25)},mode='symmetric'),
        iaa.Noop(),
        iaa.CropAndPad(percent=(-0.2, 0.2),pad_mode="reflect",keep_size=False),
    ])
    # More as you want ...
])
seq_train = iaa.Sequential(
    sometimes(iaa.Multiply((0.8,1.2))),  # , per_channel=0.5
    sometimes(iaa.Add((-0.2, 0.2))),  # , per_channel=0.5
    sometimes(iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(0, 0.05)),
        iaa.GaussianBlur(sigma=(0.0,1.0)),
    ]))
)

def make_image_gen(features, labels, batch_size=32):
    num = features.shape[0]
    all_batches_index = np.arange(0, num)
    while True:
        np.random.shuffle(all_batches_index)
        for i in range(0,num,batch_size):
            out_images, out_masks = [], []
            for index in all_batches_index[i:i+batch_size]:
                c_img, c_mask = do_augmentation(seq, seq_train, features[index], labels[index])
                out_images.append(c_img)
                out_masks.append(c_mask)
            yield np.stack(out_images, 0), np.stack(out_masks, 0)

''' ######### common operations end ############# '''
with open(path_pickle, 'rb') as f:
    mytrainer= pickle.load(f)

if not os.path.isfile(path_temp_model):
    if not os.path.isdir(direct): os.mkdir(direct)

    ''' ######### model initialization ############# '''
    #################################### model defining ##########################################
    ## ResNet helper ##
    def BatchActivate(x):
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        return x

    def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation == True:
            x = BatchActivate(x)
        return x

    def residual_block(blockInput, num_filters=16, batch_activate = False):
        x = BatchActivate(blockInput)
        x = convolution_block(x, num_filters, (3,3) )
        x = convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        if batch_activate:
            x = BatchActivate(x)
        return x

    def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
        # 128 -> 64 #101 -> 50
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
        conv1 = residual_block(conv1,start_neurons * 1)
        conv1 = residual_block(conv1,start_neurons * 1, True)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(DropoutRatio/2)(pool1)

        # 64 -> 32 #50 -> 25
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
        conv2 = residual_block(conv2,start_neurons * 2)
        conv2 = residual_block(conv2,start_neurons * 2, True)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(DropoutRatio)(pool2)

        # 32 -> 16 #25 -> 12
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
        conv3 = residual_block(conv3,start_neurons * 4)
        conv3 = residual_block(conv3,start_neurons * 4, True)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(DropoutRatio)(pool3)

        # 16 -> 8 #12 -> 6
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
        conv4 = residual_block(conv4,start_neurons * 8)
        conv4 = residual_block(conv4,start_neurons * 8, True)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(DropoutRatio)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
        convm = residual_block(convm,start_neurons * 16)
        convm = residual_block(convm,start_neurons * 16, True)

        # 8 -> 16  #6 -> 12
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(DropoutRatio)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4,start_neurons * 8)
        uconv4 = residual_block(uconv4,start_neurons * 8, True)

        # 16 -> 32 #12 -> 25
        # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(DropoutRatio)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3,start_neurons * 4)
        uconv3 = residual_block(uconv3,start_neurons * 4, True)

        # 32 -> 64 #25 -> 50
        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = residual_block(uconv2,start_neurons * 2)
        uconv2 = residual_block(uconv2,start_neurons * 2, True)

        # 64 -> 128 #50 -> 101
        # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(DropoutRatio)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = residual_block(uconv1,start_neurons * 1)
        uconv1 = residual_block(uconv1,start_neurons * 1, True)

        # uconv1 = Dropout(DropoutRatio)(uconv1)
        # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
        output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
        output_layer =  Activation('sigmoid')(output_layer_noActi)

        return output_layer

    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16)

    c = optimizers.adam(lr=1e-2)
    model = Model(input_layer, output_layer)
    model.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
    # model.summary()
    ################################ model defining finised ######################################
    ''' ######### model initialization end ############# '''
else:
    model = mytrainer.load(path_temp_model)

(X, y, train_all, evaluate_all) = [np.load(file) for file in files]
train_index,evaluate_index  = train_all[cv_ind],evaluate_all[cv_ind]
x_train,y_train = X[0,train_index],y[0,train_index]
x_valid,y_valid = X[0,evaluate_index],y[0,evaluate_index]

if not mytrainer.isStopped():
    ## data augmentation with left-right flipping ##
    num = x_train.shape[0]
    steps = num*2//batch_size +(num*2%batch_size!=0)
    train_generator = make_image_gen(x_train, y_train, batch_size)
    mytrainer.train_generator(model,path_temp_model,train_generator,steps_per_epoch= steps,validation_data=[x_valid, y_valid], verbose=2)
    etime = time.time()
    print("time consumed:" + str(etime-stime))
    exit()

if  os.path.isfile(path_sub):
    print("already finished.")
    exit()

''' ######### make prediction ############# '''
model = mytrainer.load(path_best_model)
preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)

## IoU metrics with threshold -- copied -- could revise ##
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
    
## find best threshold ##
thresholds = np.linspace(0, 1, 100)
ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])
threshold_best_index = np.argmax(ious) # np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
np.savetxt(path_thres,np.c_[thresholds,ious])

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
x_test = np.array([np.array(load_img(path_input+"test/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
preds_test = model.predict(x_test)
pred_dict = {idx: RLenc(np.round(preds_test[i]) > threshold_best) for i, idx in enumerate(tqdm(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(path_sub)

etime = time.time()
print("time consumed:" + str(etime-stime))
print(threshold_best_index,threshold_best,iou_best)
