"""
As is common for fMRI data, we train a simple
linear SVM on masked training data to classify the masked
test data. By applying the computed mask to a brain volume,
invalid voxels are discarded and valid voxels are placed in
a 1-dimensional vector, thus reducing the dimension of the
brain volumes. The mask is computed based on the training
data. Several strategies of mask computation can be used,
e.g., computing the mask corresponding to gray matter part of
the brain or computing the mask of the background from the
image border. We conduct the experiments on several masking
strategies and do not find much difference

- to try, mask from border. vs. mask from gray matter to reduce dimensionality, get rid of unwanted data
"""


# SVM
import os
import random
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score

n_synthetic = [2, 10, 50, 100, 150, 200, 300, 500]
n_real = 500 # maximum is 2016
from Dataset import Mixed, RealTest
import time
import matplotlib
matplotlib.use("Agg")



path = 'C:/Users/hlw69/Documents/fMRI_transferlearning/Zhang et al/ICW-GANs-master/ICW-GANs-master/'
synth_dir = path + 'tests/'
def load_real_data(task = 'perceive'):
    # load up all the real ones  - batch_generator_for_classify or new_load_data from brainpedia
    real_labels = np.load(path + 'tflib/test_data/'+task+'/labels.npy') # take second, as in the ID used for saving, it is this second index which is stored in the name, assume this is the relevant class
    print("Before filtering ,real labels are ", real_labels.shape)
    real_labels = [x[1] for x in real_labels]
    print("How many real labels ", len(real_labels))
    real_data = np.load(path +'tflib/test_data/'+task+'/data.npy' )
    return real_data, np.asarray(real_labels)

def load_synthetic_data(amount):
    synthetic_data = []
    synthetic_label = []
    for file in os.listdir(path+'synthetic'):
       # print(file)
        img = nib.load(path+'synthetic/'+file)
        data = img.get_fdata()
        synthetic_data.append(data)
        stripped = file.replace('.nii.gz',"")
        label = int(stripped[-1])
      #  print(label)
        synthetic_label.append(label)
        # get labels associated
   # synthetic_data = path + 'tests/'  # load up all .nii.gz and convert to .npy format
    x_sub, y_sub = zip(*random.sample(list(zip(synthetic_data, synthetic_label)), amount))
    return x_sub, np.asarray(list(y_sub))

# amount has to be bigger than zero for the code to run9
def svm_classifier(condition='both', amount=5, test = 'real', task = 'perceive', noise = False):
    print('The data is ', condition, ' for ', task, ' task, is noise added?', noise)
    # load real data
    all_real_x, all_real_y = load_real_data(task)
    # train and test split for real data
    X_train, X_test, y_train_real, y_test = train_test_split(all_real_x, all_real_y, test_size=0.2, random_state=42)
    X_train = X_train[:n_real, :, :, :]
    y_train_real = y_train_real[:n_real]
    # merge the synthetic and real training data and labels together, randomly shuffle
    xyz = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    if condition != 'real':
        synth_x, synth_y = load_synthetic_data(amount)
        mixed_data = np.concatenate((synth_x,  X_train), axis=0)
        mixed_labels = np.concatenate((synth_y, y_train_real), axis=0)
        shuffled_mixed_data, shuffled_mixed_labels = shuffle(mixed_data, mixed_labels)
        # preprocessing, flatten data arrays from 4d to 2D (necerssary to be accepted by SVM)
        synth_flat = np.reshape(synth_x, (synth_x.shape[0], xyz))
        mixed_training_2d = np.reshape(shuffled_mixed_data, (shuffled_mixed_data.shape[0], xyz)) # SYNTHETIC
    X_test_2d_real = np.reshape(X_test,(X_test.shape[0],xyz)) # REAL
    X_train_2d_real = np.reshape(X_train,(X_train.shape[0],xyz))
    real_all_2d_data = np.reshape(all_real_x,(all_real_x.shape[0],xyz))
    clf = SVC(kernel='linear', C=100)
    if condition == 'both':
        print("Using a mixture of ", X_train.shape[0], " real and ", amount, " synthetic data")
        #  Both: synthetic and real data for train, and just real for test
        x_train = mixed_training_2d
        y_train = shuffled_mixed_labels
        # y test already defined outside function
    elif condition == 'real':
        # Real: all real data with a train and test split
        print("using only real data")
        x_train = X_train_2d_real
        y_train = y_train_real
    if noise == True:
        noisy_data = x_train[0:30] + np.random.normal(0, 0.01, x_train[0:30].shape) # mean, std, shape
        x_train = np.concatenate((x_train, noisy_data), axis=0)
        y_train = np.concatenate((y_train, y_train[0:30]), axis=0)
        X_test_2d_real = X_test_2d_real + np.random.normal(0, 0.01, X_test_2d_real.shape) # mean, std, shape
    #print("Training labels are ", y_train)
    print("Size of training data and labels are ", x_train.shape, y_train.shape)
    print(y_train)
        # y train and test are already defined outside this function
    #print(twod_mixed_training_data.shape)
    clf.fit(x_train, y_train)
    if test == 'real':
        pred = clf.predict(X_test_2d_real)  # always test on real data
    #elif test == 'synth':
        #pred = clf.predict(X_test_2d_real)  # always test on real data
    #print(pred, y_test)
    print("Accuracy: ", accuracy_score(y_test, pred), ". F1: ", f1_score(y_test, pred, average='macro'),
          ". Precision: ", precision_score(y_test, pred),
          ". Recall: ", recall_score(y_test, pred))
    # should cross validation take the concatenated version?
    x, y = shuffle(np.concatenate((x_train,X_test_2d_real), axis=0), np.concatenate((y_train, y_test), axis=0))
    cross_v_scores = cross_val_score(clf, x,y, cv=3) # not accurate as can end up testing on synthetic data
   # cross_v_scores = cross_val_score(clf, x_train, y_train, cv=3) # all data, no test train split
    print("cross validation score is ", np.mean(cross_v_scores))

#for n in n_synthetic:
#    svm_classifier('both', amount=n)


""" to test Gaussian noise with a mean value of 0 and a
variance of 0.01 as data augmentation
svm_classifier(condition='real', task = 'perceive')
svm_classifier(condition='real', task = 'imagine')
"""





# should compare these to downsampled versions
# try training on perception and testing on imagination
# try a different network

svm_classifier(condition='real', amount=5, test = 'real', task = 'perceive', noise = False)
svm_classifier(condition='real', amount=5, test = 'real', task = 'imagine', noise = False)

svm_classifier(condition='real', amount=5, test = 'real', task = 'perceive', noise = True)
svm_classifier(condition='real', amount=5, test = 'real', task = 'imagine', noise = True)

