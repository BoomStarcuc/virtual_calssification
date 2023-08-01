# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import json

def load_attributes(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["image_path", "fea1", "fea2", "fea3", "fea4", "fea5", "fea6", "fea7", "fea8", "fea9", "fea10", "fea11", "fea12", "fea13", "fea14", "fea15", "fea16", "fea17", "GT_label"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
    # print(df.dtypes)
    
    #data augmentation
    df_l1 = pd.DataFrame(np.repeat(df[df["GT_label"] == 1].values, 20, axis = 0), columns=cols).astype(df.dtypes.to_dict())
    df_l2 = pd.DataFrame(np.repeat(df[df["GT_label"] == 2].values, 1, axis = 0), columns=cols).astype(df.dtypes.to_dict())
    df_l3 = pd.DataFrame(np.repeat(df[df["GT_label"] == 3].values, 1, axis = 0), columns=cols).astype(df.dtypes.to_dict())
    df_l4 = pd.DataFrame(np.repeat(df[df["GT_label"] == 4].values, 2, axis = 0), columns=cols).astype(df.dtypes.to_dict())
    # print(df.values)
    # print("1\n",df_l1)
    # print("2\n",df_l2)
    # print("3\n",df_l3)
    # print("4\n",df_l4)

    df = pd.concat([df_l1, df_l2, df_l3, df_l4])
    # print(df.dtypes)
    
    # determine (1) the unique zip codes and (2) the number of data
    # points with each zip code
    # zipcodes = df["zipcode"].value_counts().keys().tolist()
    # counts = df["zipcode"].value_counts().tolist()
    # # loop over each of the unique zip codes and their corresponding
    # # count
    # for (zipcode, count) in zip(zipcodes, counts):
    #     # the zip code counts for our housing dataset is *extremely*
    #     # unbalanced (some only having 1 or 2 houses per zip code)
    #     # so let's sanitize our data by removing any houses with less
    #     # than 25 houses per zip code
    #     if count < 25:
    #         idxs = df[df["zipcode"] == zipcode].index
    #         df.drop(idxs, inplace=True)
    # # return the data frame
    return df

def process_attributes(train, test):
    # initialize the column names of the continuous data
    continuous = ["fea1", "fea2", "fea3", "fea4", "fea5", "fea6", "fea7", "fea8", "fea9", "fea10", "fea11", "fea12", "fea13", "fea14", "fea15", "fea16", "fea17"]
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # trainContinuous = train[continuous]
    # testContinuous = test[continuous]

    # print("trainF1:", trainContinuous)
    # print("testF1:", testContinuous)
    
    # # one-hot encode the zip code categorical data (by definition of
    # # one-hot encoding, all output features are now in the range [0, 1])
    # zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    # trainCategorical = zipBinarizer.transform(train["zipcode"])
    # testCategorical = zipBinarizer.transform(test["zipcode"])
    # # construct our training and testing data points by concatenating
    # # the categorical features with the continuous features
    trainX = np.hstack([trainContinuous, np.expand_dims(np.array(train['GT_label']), axis=1)])
    testX = np.hstack([testContinuous, np.expand_dims(np.array(test['GT_label']), axis=1)])
    # trainX = np.hstack([trainCategorical, trainContinuous])
    # testX = np.hstack([testCategorical, testContinuous])
    # return the concatenated training and testing data
    # trainX = trainContinuous
    # testX = testContinuous
    return (trainX, testX)

def process_testset(test):
    continuous = ["fea1", "fea2", "fea3", "fea4", "fea5", "fea6", "fea7", "fea8", "fea9", "fea10", "fea11", "fea12", "fea13", "fea14", "fea15", "fea16", "fea17"]
    
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(test[continuous])
    testX = np.hstack([testContinuous, np.expand_dims(np.array(test['GT_label']), axis=1)])
    return testX

def load_images(df, inputPath):
    # initialize our images array (i.e., the house images themselves)
    images = []
    # loop over the indexes of the houses
    for i in df.image_path:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        if os.path.exists(os.path.sep.join([inputPath, i.strip()])):
            basePath = os.path.sep.join([inputPath, i.strip()])
        
            image = cv2.imread(basePath)
            image = cv2.resize(image, (108, 240)) # width:108, height:240 
            # cv2.imwrite("{}".format(i), image)
            images.append(image)
    # return our set of images
    print("number of df:", len(df))
    print("number of images:", len(images))
    assert len(df) == len(images)
    return np.array(images)
