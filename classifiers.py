# HW: Have created this class to implement the classifiers mentioned in their paper

"""
We compare the SVM and the 3D
deep net classifier trained with real brain images (‘Real’) or
real plus synthetic brain images (‘Real+Synth.’). Accuracy,
macro F1, precision, and recall metrics are used to measure
the results. Note that the test data for classification is always composed of real images.
Size:
 1951 = Training: real data, 100 real data, with additive Gaussian noise per class, or real data plus 100 generated data per class
 2138/BRAIN =  real data, real data plus 20 real data with additive Gaussian noise per class, or real data plus 20 generative data
per class
stratified 3-fold cross validation

"""
import os
import random
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from Dataset import Mixed, RealTest, SyntheticOnly
import torch, torchvision
from torch.nn import Conv2d, Module, Linear, Flatten
from torch.utils.data.dataset import random_split
import time
import matplotlib
matplotlib.use("Agg")



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
path = 'C:/Users/hlw69/Documents/fMRI_transferlearning/Zhang et al/ICW-GANs-master/ICW-GANs-master/'
n_synthetic = 1000
n_real = 50
def load_real_data():
    # load up all the real ones  - batch_generator_for_classify or new_load_data from brainpedia
    real_labels = np.load(path + 'tflib/test_data/labels.npy') # take second, as in the ID used for saving, it is this second index which is stored in the name, assume this is the relevant class
    real_labels = [x[1] for x in real_labels]
    real_data = np.load(path +'tflib/test_data/data.npy' )
    return real_data, np.asarray(real_labels)
def load_synthetic_data(amount = n_synthetic):
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

def svm_classifier_synth():
    synth_x, synth_y = load_synthetic_data(amount=n_synthetic)
    synth_x = np.asarray(synth_x)
    xyz = synth_x.shape[1] * synth_x.shape[2] * synth_x.shape[3]
    # print("x y z", xyz)
    synth_flat = np.reshape(synth_x, (synth_x.shape[0], xyz))
    clf = SVC(kernel='linear', C=100)
    print("using only synthetic data")
    x_train, x_test, y_train, y_test = train_test_split(synth_flat, synth_y, test_size=0.2, random_state=42)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)  # always test on real data
    print("F1: ", f1_score(y_test, pred, average='macro'),
          ". Precision: ", precision_score(y_test, pred),
          ". Recall: ", recall_score(y_test, pred))
    # should cross validation take the concatenated version?
    x, y = shuffle(np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0))
    cross_v_scores = cross_val_score(clf, x, y, cv=3)
    # cross_v_scores = cross_val_score(clf, x_train, y_train, cv=3) # all data, no test train split
    print("cross validation score is ", np.mean(cross_v_scores))


def svm_classifier(condition='mixed'):
    # load real data
    all_real_x, all_real_y = load_real_data()
    # train and test split for real data
    X_train, X_test, y_train_real, y_test = train_test_split(all_real_x, all_real_y, test_size=0.2, random_state=42)
    X_train = X_train[:n_real, :, :, :]
    y_train_real = y_train_real[:n_real]
    synth_x, synth_y = load_synthetic_data(amount=n_synthetic)
    # merge the synthetic and real training data and labels together, randomly shuffle
    mixed_data = np.concatenate((synth_x,  X_train), axis=0)
    mixed_labels = np.concatenate((synth_y, y_train_real), axis=0)
    shuffled_mixed_data, shuffled_mixed_labels = shuffle(mixed_data, mixed_labels)
    #print(shuffled_mixed_data.shape,  shuffled_mixed_labels.shape)
    # train SVM classifier with cross validation
    # preprocessing, flatten data arrays from 4d to 2D (necerssary to be accepted by SVM)
    synth_x = np.asarray(synth_x)
    xyz =  synth_x.shape[1] * synth_x.shape[2] * synth_x.shape[3]
    #print("x y z", xyz)
    synth_flat = np.reshape(synth_x, (synth_x.shape[0], xyz))
    mixed_training_2d = np.reshape(shuffled_mixed_data, (shuffled_mixed_data.shape[0], xyz)) # SYNTHETIC
    X_test_2d_real = np.reshape(X_test,(X_test.shape[0],xyz)) # REAL
    X_train_2d_real = np.reshape(X_train,(X_train.shape[0],xyz))
    real_all_2d_data = np.reshape(all_real_x,(all_real_x.shape[0],xyz))
    clf = SVC(kernel='linear', C=100)
    if condition == 'mixed':
        print("Using a mixture of real and ", n_synthetic, " synthetic data")
        #  Both: synthetic and real data for train, and just real for test
        x_train = mixed_training_2d
        y_train = shuffled_mixed_labels
        # y test already defined outside function
    elif condition == 'real':
        # Real: all real data with a train and test split
        print("using only real data")
        x_train = X_train_2d_real
        y_train = y_train_real
    #print("Training labels are ", y_train)
    print("Size of training data and labels are ", x_train.shape, y_train.shape)
        # y train and test are already defined outside this function
    #print(twod_mixed_training_data.shape)
    clf.fit(x_train, y_train)

    pred = clf.predict(X_test_2d_real) # always test on real data
    #print(pred, y_test)
    print("F1: ", f1_score(y_test, pred, average='macro'),
          ". Precision: ", precision_score(y_test, pred),
          ". Recall: ", recall_score(y_test, pred))
    # should cross validation take the concatenated version?
    x, y = shuffle(np.concatenate((x_train,X_test_2d_real), axis=0), np.concatenate((y_train, y_test), axis=0))
    cross_v_scores = cross_val_score(clf, x,y, cv=3)
   # cross_v_scores = cross_val_score(clf, x_train, y_train, cv=3) # all data, no test train split
    print("cross validation score is ", np.mean(cross_v_scores))


svm_classifier(condition='mixed')
svm_classifier(condition ='real')











"""
The deep neural net structure is similar to the discriminator with a 3-dimensional structure and
an identical number of convolution layers with Leaky ReLU
activations. Unlike the discriminator, the classifier obviously
does not concatenate intermediate and input data with any
label information. Need to add in batchNorm, they do not seem to do pooling.
"""
torch.multiprocessing.freeze_support()
path = 'C:/Users/hlw69/Documents/fMRI_transferlearning/Zhang et al/ICW-GANs-master/ICW-GANs-master/'
synth_dir = path + 'synthetic/'
def load_real_data():
    # load up all the real ones  - batch_generator_for_classify or new_load_data from brainpedia
    real_labels = np.load(path + 'tflib/test_data/labels.npy') # take second, as in the ID used for saving, it is this second index which is stored in the name, assume this is the relevant class
    real_labels = [x[1] for x in real_labels] # takes the second index, as this should correspond to saved synthetic
    real_data = np.load(path +'tflib/test_data/data.npy' )
    return real_data, np.asarray(real_labels)
# load real data
all_real_x, all_real_y = load_real_data()
# train and test split for real data
X_train, X_test, y_train_real, y_test = train_test_split(all_real_x, all_real_y, test_size=0.2, random_state=42)
# validation(?)
n_synthetic = 1000
batch_size = 32
INIT_LR = 0.01
EPOCHS = 500
condition = 'synthetic_only' # could be Mixed

# %TODO: Make a dataset loader that is only synthetic to divide into train and test and validation
condition = 'synthetic_only' # could be Mixed
# GET TRAIN, VAL AND TEST DATA LOADED UP
if condition == 'mixed':
    synthetic_dataset = Mixed(synth_dir, X_train, y_train_real, transform=None, is_test=False, amount=n_synthetic)
    train_n = int(0.7 * len(synthetic_dataset))
    val_n = len(synthetic_dataset) - train_n
    trainset, valset = random_split(synthetic_dataset, (train_n, val_n))
    testset = RealTest(X_test, y_test, transform=None, is_test=True)
elif condition == 'synthetic_only':
    synthetic_dataset = SyntheticOnly(synth_dir, transform=None, is_test=False, amount=n_synthetic)
    train_n = int(0.8 * len(synthetic_dataset))
    test_val_n = len(synthetic_dataset) - train_n
    trainset, val_and_test_set = random_split(synthetic_dataset, (train_n, test_val_n))
    test_n = int(0.5 *len(val_and_test_set))
    val_n = len(val_and_test_set) - test_n
    testset, valset = random_split(val_and_test_set, (test_n, val_n))

else:
    print("An incorrect choice of dataset has been provided")





trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


class DNN(Module):
    def __init__(self, numChannels = 53, classes=2):
        # call the parent constructor
        super(DNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=46, out_channels=24,
                kernel_size=4, stride=2, padding=0)
    # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=24, out_channels=12,
			kernel_size= 4, stride=2,padding=1)
        self.conv3 = Conv2d(in_channels=12, out_channels=5,
			kernel_size= 4, stride=2,padding=1)
        self.conv4 = Conv2d(in_channels=5, out_channels=4,
			kernel_size= 4, stride=2,padding=1)

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = torch.nn.Linear(in_features=36, out_features=2)
        # initialize our softmax classifier
        self.classify = torch.nn.Sigmoid()
    def forward(self,x):
        x = np.swapaxes(x, 1,3) # channels should come before height and width
        print("Shape of input is ", x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        print("Shape of x is ", x.shape)
       # x = Flatten(x)
        x = self.fc1(x)

        output = self.classify(x)
        return output

#https://www.pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# calculate steps per epoch for training and validation set
trainSteps = len(trainloader.dataset) // batch_size
valSteps = len(valloader.dataset) // batch_size
print("[INFO] initializing the DNN model...")

model = DNN(
	numChannels=46, classes=2).to(device).float()
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
lossFn = torch.nn.NLLLoss()
# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

for e in range(0, EPOCHS):
    print("Running epoch ", e)
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (x, y) in trainloader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x.float())
        loss = lossFn(pred, y.long())
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.double).sum().item()
# switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        for (x, y) in valloader:         # loop over the validation set
            (x, y) = (x.to(device), y.to(device)) 			# send the input to the device
            pred = model(x.float()) 			# make the predictions and calculate the validation loss
            totalValLoss += lossFn(pred, y)
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(
				torch.double).sum().item()

    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainloader.dataset)
    valCorrect = valCorrect / len(valloader.dataset)
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
   # print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in testloader:
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
# generate a classification report
print(classification_report(testset.targets.cpu().numpy(),
                            np.array(preds), target_names=testset.classes))

import matplotlib as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show(0)
#plt.savefig(args["plot"])
# serialize the model to disk
#torch.save(model, args["model"])





