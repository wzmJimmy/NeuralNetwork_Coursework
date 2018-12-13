import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import load_img
from tqdm import tqdm
import math
import time
import os


direct = "../training/"
path_input= "../../input/"
path_train = direct+"train_df.csv"
path_test_depth = direct+"depth.csv"
files = [direct+i+'.npy' for i in ['X','Y', 'trains','valids']]

seed = 1214
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)

batch_size = 32
cv_total = 5
cv_ind = 0
img_size_ori = 101
img_size_target = 101 #128

stime = time.time()

if not os.path.isfile(path_test_depth):
    if not os.path.isdir(direct): os.mkdir(direct)

    ''' train and evaluation data initialization  '''
    ## loading train dataset ##
    train_df = pd.read_csv(path_input + "train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv(path_input + "depths.csv", index_col="id")
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    test_df.to_csv(path_test_depth)

    train_df = train_df.join(depths_df)
    image, mask = [], []
    for idx in tqdm(train_df.index):
        image.append(np.array(load_img(path_input + "train/images/{}.png".format(idx), grayscale=True)) / 255)
        mask.append(np.array(load_img(path_input + "train/masks/{}.png".format(idx), grayscale=True)) / 255)
    train_df["images"] = image
    train_df["masks"] = mask
    ## find strata ##
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(lambda val: math.ceil(val * 10))

    X = np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1),
    y = np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1),

    train_all,evaluate_all = [],[]
    skf = StratifiedKFold(n_splits=cv_total, random_state=seed, shuffle=True)
    for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
        train_all.append(train_index)
        evaluate_all.append(evaluate_index)
        print(train_index.shape, evaluate_index.shape)

    datas = [X, y, train_all, evaluate_all]
    for i, file in enumerate(files):
        np.save(file, datas[i])

etime = time.time()
print("time consumed:" + str(etime - stime))