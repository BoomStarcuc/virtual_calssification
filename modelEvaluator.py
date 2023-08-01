# import the necessary packages
from model import datasets
from model import models
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
import numpy as np
import argparse
import locale
import os
import matplotlib.pylab as plt
import random
import time
import shutil 

testsetSelectPath = "data/shuffledDataList.txt"
inputPath = "data/format_each_allDatas.txt"
imagePath = "image"
dataPath = "data"
bugPath = 'bugs'
tsPath = 'testSets'
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"
# need to change some paths and variables according to which dataset, to include some errors
# selector
best_model_filepath = "chosenModel/checkpoint_82.91acc_K4_dataset9/best_model_F4.h5"


def main():
    seed_value= 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of images")
    #args = vars(ap.parse_args())# construct the path to the input .txt file that contains information
    # on each house in the dataset and then load the dataset
    print("[INFO] loading attributes...")
    
    df = datasets.load_attributes(inputPath)
    selector = int(best_model_filepath.split('/')[1].split('_')[-1][-1])
    print("[INFO] The selector is:", selector)
    trainset, testset = splitDataSet(testsetSelectPath, selector)
    #print(len(flatten(trainset.values())))
    #print(len(flatten(testset.values())))
   
    df_test = df[df["image_path"].isin(testset)]
    df = df[df["image_path"].isin(trainset)]
    #print(len(df))
    #print(len(df_test))
    print('[INFO] dataset', selector, 'has total data entry', len(df_test))
    

    # load the house images and then scale the pixel intensities to the
    # range [0, 1]
    print("[INFO] loading images...")
    # split the data into train set and test set, store the data into file for fast load later
    # Randomly select the test set, store the file into different folder by different selector(related to random seed)
    # Generate the dir if not exist
    processedImagePath = os.path.join(dataPath, str(selector), processed_images_filename)
    if not os.path.exists(os.path.join(dataPath, str(selector))):
        os.makedir(os.path.join(dataPath, str(selector)))
    
    if os.path.exists(processedImagePath):
        images = np.load(processedImagePath)
    else:
        images = datasets.load_images(df, imagePath)
        images = images / 255.0
        np.save(processedImagePath, images)
        
    processedImageTestPath  = os.path.join(dataPath, str(selector), processed_images_test_filename)
    if os.path.exists(processedImageTestPath):
        images_test = np.load(processedImageTestPath)
    else:
        images_test = datasets.load_images(df_test, imagePath)
        images_test = images_test / 255.0
        np.save(processedImageTestPath, images_test)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    print("[INFO] processing data...")
    
    # Prepare the test set
    testSetImagesX = images_test
    testSetAttrX = df_test
    testSetY = np.array(testSetAttrX["GT_label"]) - 1
    testSetY = to_categorical(testSetY, num_classes=4)
    testSetAttrX = datasets.process_testset(testSetAttrX)

    #Load and evaluate the best model version
    print("[INFO] best model is {}".format(best_model_filepath))
    model = load_model(best_model_filepath, custom_objects={'ResnetBlock': models.ResnetBlock})
    #model = load_model(best_model_filepath)

    # make predictions on the testing data
    print("[INFO] computing accuracy...")
    preds = model.predict([testSetAttrX, testSetImagesX])
    neg = 0
    resPred = np.argmax(preds, axis=1)
    print(resPred)
    resGT = np.argmax(testSetY, axis=1)
    print(resGT)
    #print(testSetY)
    
    # Put all bug image together in /bugs
    # Remove all existing images in /bugs 
    for f in os.listdir(bugPath):
        os.remove(os.path.join(bugPath, f))
    for f in os.listdir(tsPath):
        os.remove(os.path.join(tsPath, f))
    
        
    # Calculate acc for each object
    record = {}
    total_nodup = 0
    correct_nodup = 0
    for i in range(0, len(preds)):
        pred = resPred[i]
        gt = resGT[i]
        img = df_test.iloc[i]['image_path']
        if img in record:
            continue
        if gt == 0 or gt == 1:
            shutil.copy(os.path.join(imagePath, img), os.path.join(tsPath, img))
        total_nodup = total_nodup + 1
        if pred == gt:
            correct_nodup = correct_nodup + 1
            if pred == 0 or pred == 1:
                shutil.copy(os.path.join(imagePath, img), os.path.join(bugPath, img))
            record[img] = True
            continue
        record[img] = False
    
    assert total_nodup == len(record)
 
    print("[INFO] Accuracy No Dup: %s" % "{:.2f}%".format(correct_nodup / total_nodup * 100))
    
    print("[INFO] another accuracy: {:.2f}%".format(np.sum(resPred == resGT) / len(preds) * 100))     
    #print(preds)
    (eval_loss, eval_accuracy) = model.evaluate( 
        [testSetAttrX, testSetImagesX], testSetY, batch_size=32, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss))
    
def splitDataSet(path, selector):
    total = []
    trainset = []
    testset = []
    
    with open(path) as f:
        for line in f:
            img = line.rstrip('\n')
            total.append(img)
            
    count = 0
    ratio = 0.1
    numTestObj = int(len(total) * ratio)
    print("[INFO] Number of objects in test set", numTestObj)
    interval = int(1 / ratio)
    
    for img in total:
        if count % interval == selector and len(testset) < numTestObj:
            testset.append(img)
        else:
            trainset.append(img)
        count = count + 1
        
    assert len(testset) == numTestObj
    assert len(trainset) + len(testset) == len(total)
    return trainset, testset

def flatten(l):
    return [item for sublist in l for item in sublist]
    

if __name__ == "__main__":
    #start_time = time.time()
    main()
    #print("--- Execution time: %s seconds ---" % (time.time() - start_time))