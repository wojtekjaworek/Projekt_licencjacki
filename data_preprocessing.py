from tda_pipelines import *

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.util import random_noise
from sklearn.utils import shuffle



# load pipelines 


tda_pipeline_34 = TDA_PI34_Pipeline()
tda_pipeline_42 = TDA_PI42_Pipeline()
vector_stitching_pipeline_34, tda_union_34 = VECTOR_STITCHING_PI_Pipeline_34()
vector_stitching_pipeline_42, tda_union_42 = VECTOR_STITCHING_PI_Pipeline_42()


def process_mnist_alike(data = (None,None), training_indices=(1,1), testing_indices=(1,1), dist_ratio=10, clean=True, distorted=True):
    """
    Full data processing for validation experiment. Works ONLY for 28x28 pixels and one layer images . (MNIST alike)
    
    Returns dict with keys:

        "X_tr": X_tr, # 90% clean and 10% noisy normal data
        "y_tr": y_tr, # 90% clean and 10% noisy normal data labels
        "X_tr_tda_34": X_tr_tda_34, # 90% clean and 10% noisy tda data
        "y_tr_tda_34": y_tr_tda_34, # 90% clean and 10% noisy tda data labels 
        "X_tr_vector_stitching_34": X_tr_vector_stitching_34, # 90% clean and 10% noisy vector stitching data
        "y_tr_vector_stitching_34": y_tr_vector_stitching_34, # 90% clean and 10% noisy vector stitching data labels
        "X_train": X_train_expanded,
        "y_train": y_train,
        "X_train_noisy_random": X_train_noisy_random_expanded,
        "X_train_clean_tda_good_34": X_train_clean_tda_good_34,    
        "X_train_noisy_tda_good_34": X_train_noisy_tda_good_34,
        "X_train_clean_vector_stitching_34": X_train_clean_vector_stitching_34,
        "X_train_noisy_vector_stitching_34": X_train_noisy_vector_stitching_34,
        "X_test": X_test_expanded, 
        "y_test": y_test,
        "X_test_noisy_random": X_test_noisy_random_expanded,
        "X_test_clean_tda_good_34": X_test_clean_tda_good_34,
        "X_test_noisy_tda_good_34": X_test_noisy_tda_good_34,
        "X_test_clean_vector_stitching_good_34": X_test_clean_vector_stitching_good_34,
        "X_test_noisy_vector_stitching_good_34": X_test_noisy_vector_stitching_good_34


    """

    (X, y) = data

    train_range = range(training_indices[0], training_indices[1])
    test_range = range(testing_indices[0], testing_indices[1])

    X = X.reshape((-1, 28, 28))


    X_train = X[train_range]
    X_test = X[test_range]
    y_train = y[train_range]
    y_test = y[test_range]


    # mnist comes with string labels, we need to convert them to int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


    # distort X_train and X_test a little bit not using giotto
    X_train_noisy = random_noise(X_train, mode="s&p",amount=0.05, seed=666)
    X_test_noisy = random_noise(X_test, mode="s&p",amount=0.05, seed=666)

    # generate random noise matrix of size X_train_noisy.shape and X_test_noisy.shape but without original image

    X_train_gaussian_noise = np.random.rand(*X_train_noisy.shape)
    X_test_gaussian_noise = np.random.rand(*X_test_noisy.shape)

    # for each image in X_train_noisy and X_test_noisy, we will add the random noise matrix to the image

    X_train_noisy_random = X_train_noisy + X_train_gaussian_noise
    X_test_noisy_random = X_test_noisy + 0.5*X_test_gaussian_noise




    # clean data
    X_train_clean_tda_34 = tda_pipeline_34.fit_transform(X_train)
    X_test_clean_tda_34 = tda_pipeline_34.transform(X_test)
    print("1/4")
    

    
    # distorted data
    X_train_noisy_tda_34 = tda_pipeline_34.fit_transform(X_train_noisy_random)
    X_test_noisy_tda_34 = tda_pipeline_34.transform(X_test_noisy_random)
    print("2/4")

    # generate TDA features

    #clean data
    X_train_clean_vector_stitching_34 = vector_stitching_pipeline_34.fit_transform(X_train)
    X_test_clean_vector_stitching_34 = vector_stitching_pipeline_34.transform(X_test)
    print("3/4")

    # distorted data
    X_train_noisy_vector_stitching_34 = vector_stitching_pipeline_34.fit_transform(X_train_noisy_random)
    X_test_noisy_vector_stitching_34 = vector_stitching_pipeline_34.transform(X_test_noisy_random)
    print("4/4")

    # tda
    X_train_clean_tda_good_34 = np.transpose(X_train_clean_tda_34, (0, 3, 2, 1))
    X_test_clean_tda_good_34 = np.transpose(X_test_clean_tda_34, (0, 3, 2, 1))

    X_train_noisy_tda_good_34 = np.transpose(X_train_noisy_tda_34, (0, 3, 2, 1))
    X_test_noisy_tda_good_34 = np.transpose(X_test_noisy_tda_34, (0, 3, 2, 1))

    #stitched
    X_train_clean_vector_stitching_good_34 = np.transpose(X_train_clean_vector_stitching_34, (0, 3, 2, 1))
    X_test_clean_vector_stitching_good_34 = np.transpose(X_test_clean_vector_stitching_34, (0, 3, 2, 1))

    X_train_noisy_vector_stitching_good_34 = np.transpose(X_train_noisy_vector_stitching_34, (0, 3, 2, 1))
    X_test_noisy_vector_stitching_good_34 = np.transpose(X_test_noisy_vector_stitching_34, (0, 3, 2, 1))


    #fix dimensions
    X_train_expanded, X_test_noisy_random_expanded, X_test_expanded = transform_data(X_train, X_test_noisy_random, X_test)
    _, X_train_noisy_random_expanded, _ = transform_data(X_train, X_train_noisy_random, X_test)





    # prepare 90% clean and 10% noisy data for training and testing (or any other ratio)

    percent = dist_ratio

    # calculate how many distorted data should be added to the clean data
    n = int((percent/100)*X_train_expanded.shape[0])


    X_tr = np.concatenate((X_train_expanded[:-n], X_train_noisy_random_expanded[:n]), axis=0)
    y_tr = np.concatenate((y_train[:-n], y_train[:n]), axis=0)
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=666)

    # the same for other data
    X_tr_tda_34 = np.concatenate((X_train_clean_tda_good_34[:-n], X_train_noisy_tda_good_34[:n]), axis=0)
    y_tr_tda_34 = np.concatenate((y_train[:-n], y_train[:n]), axis=0)
    X_tr_tda_34, y_tr_tda_34 = shuffle(X_tr_tda_34, y_tr_tda_34, random_state=666)


    X_tr_vector_stitching_34 = np.concatenate((X_train_clean_vector_stitching_good_34[:-n], X_train_noisy_vector_stitching_good_34[:n]), axis=0)
    y_tr_vector_stitching_34 = np.concatenate((y_train[:-n], y_train[:n]), axis=0)
    X_tr_vector_stitching_34, y_tr_vector_stitching_34 = shuffle(X_tr_vector_stitching_34, y_tr_vector_stitching_34, random_state=666)

    output = {
        "X_tr": X_tr, # 90% clean and 10% noisy normal data
        "y_tr": y_tr, # 90% clean and 10% noisy normal data labels
        "X_tr_tda_34": X_tr_tda_34, # 90% clean and 10% noisy tda data
        "y_tr_tda_34": y_tr_tda_34, # 90% clean and 10% noisy tda data labels 
        "X_tr_vector_stitching_34": X_tr_vector_stitching_34, # 90% clean and 10% noisy vector stitching data
        "y_tr_vector_stitching_34": y_tr_vector_stitching_34, # 90% clean and 10% noisy vector stitching data labels
        "X_train": X_train_expanded,
        "y_train": y_train,
        "X_train_noisy_random": X_train_noisy_random_expanded,
        "X_train_clean_tda_good_34": X_train_clean_tda_good_34,    
        "X_train_noisy_tda_good_34": X_train_noisy_tda_good_34,
        "X_train_clean_vector_stitching_good_34": X_train_clean_vector_stitching_good_34,
        "X_train_noisy_vector_stitching_good_34": X_train_noisy_vector_stitching_good_34,
        "X_test": X_test_expanded, 
        "y_test": y_test,
        "X_test_noisy_random": X_test_noisy_random_expanded,
        "X_test_clean_tda_good_34": X_test_clean_tda_good_34,
        "X_test_noisy_tda_good_34": X_test_noisy_tda_good_34,
        "X_test_clean_vector_stitching_good_34": X_test_clean_vector_stitching_good_34,
        "X_test_noisy_vector_stitching_good_34": X_test_noisy_vector_stitching_good_34
    }

    return output


