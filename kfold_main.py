import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold

import general_methods
import kfold_train
import kfold_test

# parameters to adjust how many batchs and how many pictures in one batch to print the picture and labels (expected and predicted) for validation
show_images_for_how_many_batch = 1
test_batch_size = 5
# parameters to adjust how many epochs for training phase in each iteration of the 10-fold cross-validation
number_of_epoch = 10
#
k_value_in_kfold = 10

path = './dataset'  # get data set (labeled with subfolders) location

# use ImageFolder to load images in folder "train", each sub-folder is a class, 3 classes in total
dataset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([  # Compose several transform methods
    transforms.Resize((32, 32)), # resize to （h,w）. If input single number, is to keep the ratio and change the shortest edge to int
    transforms.CenterCrop(32),
    transforms.ToTensor(),  # convert data type, get the same format of training set as in examples, output is [0-1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #normalize : output = (input - mean) / std, output is [-1 ~1]
]))

table_list = [] # for average accuracy, precision, recall, f1-score for the 10 iterations in 10-fold cross-validation
conf_matrix_total = torch.zeros(3, 3) # for the confusion matrix for all 10 iterations in 10-fold cross-validation

# 10-fold cross-validation evaluation (with random shuffing)
kf = KFold(n_splits=k_value_in_kfold, shuffle=True)

# use the scikit-learn library KFold for splitting our datasets, as k=10, each time the 9/10 as training set, 1/10 as test set, iterate for 10 times
for i_fold, (train_index, test_index) in enumerate(kf.split(dataset), 1): # i_fold is the number of iteration (start from 1) of k-fold cross-validation
    train = torch.utils.data.Subset(dataset, train_index) # use train_index to get 9/10 of whole dataset as training set
    test = torch.utils.data.Subset(dataset, test_index) # use test_index to get 1/10 of whole dataset as testing set

    # load training data and testing data in batchs
    train_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=True, drop_last=False)


    print('\n################## Iteration', i_fold,  'out of 10 for 10-fold Cross-Validation ##################')
    print('Dataset is split into training set with', len(train_index), 'images and testing set with', len(test_index), 'images')

    # train with 9/10 of whole dataset in this iteration
    kfold_train.kfold_train_phase(i_fold, train_loader, number_of_epoch)

    # test with 1/10 of whole dataset in this iteration
    table_for_one_iteration = kfold_test.kfold_test_phase(i_fold, test_loader, test_batch_size, show_images_for_how_many_batch, conf_matrix_total)
    table_list.append(table_for_one_iteration)

# calculate the average accuracy, precision, recall, f1-score for the 10 iterations and print table
print('\n\n################## Average measurements for all 10 iterations in 10-fold cross-validation: ##################')
general_methods.average_measurements(table_list)

# print the confusion matrix for all 10 iterations in 10-fold cross-validation
print("\nConfusion matrix for all 10 iterations in 10-fold cross-validation:\n")
print(conf_matrix_total)
general_methods.show_confusion_matrix(conf_matrix_total)
