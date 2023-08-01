# import the necessary packages
from model import datasets
from model import models
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.layers import concatenate
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import argparse
import locale
import os
import matplotlib.pylab as plt
import random
import time
import shutil 
import sys

testsetSelectPath = "data/shuffledDataList.txt"
inputPath = "data/format_each_allDatas.txt"
outputPath = "checkpoints/checkpoint"
imagePath = "image"
dataPath = "data"
processed_images_filename = "processed_images.npy"
processed_images_test_filename = "processed_images_test.npy"

def main():
    if len(sys.argv) != 2:
        print("[ERROR] Wrong num of argument, need 2 but got", len(sys.argv))
        return
    arg1 = sys.argv[1]
    if not arg1.isdigit():
        print("[ERROR] Argument is not digit")
        return
    #remove output path if exist in case of bug
    if os.path.exists(outputPath):
        try:
            shutil.rmtree(outputPath)
        except OSError as e:
            print("Error: %s : %s" % (outputPath, e.strerror))
    
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
    selector = int(arg1)
    #selector = 0
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
        os.makedirs(os.path.join(dataPath, str(selector)))
    
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
    
    #lst_lr = [2.2e-4, 2.1e-4, 2e-4, 1.9e-4, 1.8e-4]
    #for lr in lst_lr:
    #    for i in range(2):
    #        train(images, df, images_test, df_test, selector, lr, 1e-3)
    #lst_dc = [1.2e-3, 1.1e-3, 1e-3, 9e-4, 8e-4]
    #for dc in lst_dc:
    #    train(images, df, images_test, df_test, selector, 2e-4, dc)
    
    for i in range(1):
        train(images, df, images_test, df_test, selector, 2e-4, 1e-3)
    #train(images, df, images_test, df_test, selector, 2e-4, 1e-3)

"""
Train the model
        
Parameter images: Images used for training.  

Parameter df: Other data associated with images.

Optional Parameter: 

"""
def train(images, df, images_test, df_test, selector, lr = 1e-4, dc = 1e-3):
    dc_base = 200
    #dc = dc / dc_base
    # split = train_test_split(df, images, test_size=0.25, random_state=32)
    # (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
    # print("trainAttrX:", trainAttrX.shape) #(1450, 9)
    # print("testAttrX:", testAttrX) #(484, 9)
    # print("trainImagesX:", trainImagesX.shape)# (1450, 240, 108, 3)
    # print("testImagesX:", testImagesX)# (484, 240, 108, 3)
    # find the largest house price in the training set and use it to
    # scale our house prices to the range [0, 1] (will lead to better
    # training and convergence)
    Fold = 1
    max_acc = 0
    cv = KFold(n_splits=5, random_state=33, shuffle=True)
    
    # Prepare the test set
    testSetImagesX = images_test
    testSetAttrX = df_test
    testSetY = np.array(testSetAttrX["GT_label"]) - 1
    testSetY = to_categorical(testSetY, num_classes=4)
    testSetAttrX = datasets.process_testset(testSetAttrX)
    
    for train_index, test_index in cv.split(images):
        print("Fold:", Fold)
        print("train_index:", len(train_index))
        print("validation_index:", len(test_index))

        trainImagesX = images[train_index]
        testImagesX = images[test_index]
        # print("trainImagesX:", trainImagesX.shape)
        # print("testImagesX:", testImagesX.shape)

        trainAttrX = df.iloc[train_index]
        testAttrX = df.iloc[test_index]

        # print("trainAttrX:", trainAttrX)
        # print("testAttrX:", testAttrX)

        trainY = np.array(trainAttrX["GT_label"]) - 1
        testY = np.array(testAttrX["GT_label"]) - 1
        trainY = to_categorical(trainY, num_classes=4)
        testY = to_categorical(testY, num_classes=4)

        # process the house attributes data by performing min-max scaling
        # on continuous features, one-hot encoding on categorical features,
        # and then finally concatenating them together
        (trainAttrX, testAttrX) = datasets.process_attributes(trainAttrX, testAttrX)
        # print("trainAttrX:", trainAttrX.shape)
        # print("testAttrX:", testAttrX.shape)

        # create the MLP and CNN models
        mlp = models.create_mlp(trainAttrX.shape[1], regress=False)

        '''if you want to try resnet18, apply the following 2 lines'''
        cnn = models.ResNet18([2, 2, 2, 2])
        cnn = cnn.call(108, 240, 3)
        # create the input to our final set of layers as the *output* of both
        # the MLP and CNN
        combinedInput = concatenate([mlp.output, cnn.output])
        # print("combinedInput:", combinedInput.shape) #(None, 8)

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(8, activation="relu")(combinedInput)
        x = Dense(4, activation="softmax")(x)
        # our final model will accept categorical/numerical data on the MLP
        # input and images on the CNN input, outputting a single value (the
        # predicted price of the house)
        model = Model(inputs=[mlp.input, cnn.input], outputs=x)
        # compile the model using mean absolute percentage error as our loss,
        # implying that we seek to minimize the absolute percentage difference
        # between our price *predictions* and the *actual prices*
        # filepath = './checkpoint/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'
        filepath = os.path.join(outputPath, 'best_model_F{}.h5'.format(Fold))
        checkpoint = ModelCheckpoint(filepath, 
            verbose=1, 
            monitor='val_loss',
            save_best_only=True, 
            mode='min'
        ) 
        es = EarlyStopping(monitor='val_loss', mode='min', patience=3)

        opt = Adam(learning_rate=lr, decay=dc / dc_base)
        # opt = RMSprop(learning_rate=1e-4)
        # opt = SGD(learning_rate=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])

        # train the model
        print("[INFO] training model...")
        #history  = model.fit(
        #                 x=[trainAttrX, trainImagesX], y=trainY,
        #                 validation_data=([testAttrX, testImagesX], testY),
        #                 epochs=100, batch_size=32, callbacks=[es, checkpoint])

        history  = model.fit(
                        x=[trainAttrX, trainImagesX], y=trainY,
                        validation_data=([testAttrX, testImagesX], testY),
                        epochs=100, batch_size=32, callbacks=[checkpoint])

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy') 
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(outputPath, "accuracy_K{}.jpg".format(Fold)))
        plt.clf()

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss') 
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(outputPath, "loss_K{}.jpg".format(Fold)))
        plt.clf()

        if max(val_acc) > max_acc:
            best_model_filepath = filepath
            best_model = Fold
            max_acc = max(val_acc)
            #K_testAttrX = testAttrX
            #K_testImagesX = testImagesX
            #K_testY = testY
        Fold+=1

    #Load and evaluate the best model version
    print("[INFO] best model is {}".format(best_model_filepath))
    model = load_model(best_model_filepath, custom_objects={'ResnetBlock': models.ResnetBlock})

    # make predictions on the testing data
    print("[INFO] computing accuracy...")
    preds = model.predict([testSetAttrX, testSetImagesX])
    resPred = np.argmax(preds, axis=1)
    print(resPred)
    resGT = np.argmax(testSetY, axis=1)
    print(resGT)
    #print(testSetY)
    
    # just like the method name
    #getBinaryAcc(resPred, resGT)
    
    # Put all bug image together in /bugs
    # Remove all existing images in /bugs 
    #for f in os.listdir(bugPath):
    #    os.remove(os.path.join(bugPath, f))
    #for f in os.listdir(tsPath):
    #    os.remove(os.path.join(tsPath, f))
    
        
    # Calculate acc for each object
    record = {}
    total_nodup = 0
    correct_nodup = 0
    for i in range(0, len(resPred)):
        pred = resPred[i]
        gt = resGT[i]
        img = df_test.iloc[i]['image_path']
        if img in record:
            continue
        #if gt == 0 or gt == 1:
        #    shutil.copy(os.path.join(imagePath, img), os.path.join(tsPath, img))
        total_nodup = total_nodup + 1
        if pred == gt:
            correct_nodup = correct_nodup + 1
            #if pred == 0 or pred == 1:
            #    shutil.copy(os.path.join(imagePath, img), os.path.join(bugPath, img))
            record[img] = True
            continue
        record[img] = False
    
    assert total_nodup == len(record)
    
    # Count acc of each object
    #objectAcc = {}
    #for key in testset:
    #    count = 0
    #    for item in testset[key]:
    #        if record[item]:
    #            count = count + 1
    #    acc = round(count / len(testset[key]) * 100, 2)
    #    objectAcc[key] = acc
    
    #print("[INFO] printing acc for each item:")
    #for key in objectAcc:
    #    print("%s has accuracy: %s" % (key, "{:.2f}%".format(objectAcc[key])))
    print("[INFO] Accuracy No Dup: %s" % "{:.2f}%".format(correct_nodup / total_nodup * 100))
    
    #print("[INFO] another accuracy: {:.2f}%".format(np.sum(resPred == resGT) / len(preds) * 100))     
    #print(preds)
    (eval_loss, eval_accuracy) = model.evaluate( 
        [testSetAttrX, testSetImagesX], testSetY, batch_size=32, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss))
    
    # Rename the output path to auto-save the result, and label the acc in path
    newOutputPath = '%s_%sacc_%sloss_K%d_lr%s_dc%s_dataset%d' % (outputPath, 
                                                "{:.2f}%".format(correct_nodup / total_nodup * 100),
                                                "{:.3f}".format(eval_loss), 
                                                best_model, 
                                                "{:.2e}".format(lr),
                                                "{:.2e}".format(dc),
                                                selector)
    count = 1
    while os.path.exists(newOutputPath):
        newOutputPath = newOutputPath + '_%d'%(count)
        count = count + 1
    #os.rename(outputPath, newOutputPath)
    shutil.move(outputPath, newOutputPath)
    
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