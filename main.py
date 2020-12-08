import datetime

import kfold
import test_phase

data_path = './dataset'  # get data set (labeled with subfolders) location


'''K-Fold Cross Validation'''
# parameters to adjust how many batchs and how many pictures in one batch to print the picture and labels (expected and predicted) for validation
kfold_show_images_for_how_many_batch = 2
kfold_test_batch_size = 5
# parameters to adjust how many epochs for training phase in each iteration of the 10-fold cross-validation
k_fold_number_of_epoch = 10
k_value_in_kfold = 10

start_time = datetime.datetime.now()

kfold.kfold(kfold_show_images_for_how_many_batch, kfold_test_batch_size, k_fold_number_of_epoch, k_value_in_kfold, data_path)

finish_kfold_time = datetime.datetime.now()
duration_train = finish_kfold_time - start_time
kfold_str = str(k_value_in_kfold) + '-Fold'
print('\n', kfold_str, 'Cross Validation on 80% data takes:', duration_train, '\n')
'''end of K-Fold Cross Validation'''

'''Test phase'''
test_show_images_for_how_many_batch = 0
test_batch_size = 5

print('&&&&&&&&&& Begin test using result from K-Fold Cross Validation... &&&&&&&&&&')
# test our Convolutional Neural Network with result from K-Fold Cross Validation
test_phase.test_phase(test_show_images_for_how_many_batch, test_batch_size, data_path)

finish_test_time = datetime.datetime.now()
duration_test = finish_test_time - finish_kfold_time
print('Using the result from', kfold_str, 'Cross Validation, testing phase takes:', duration_test, '\n')
'''end Test phase'''






# '''old main for project part 1'''
# import datetime
# import train_phase
# import test_phase
#
#
# test_show_images_for_how_many_batch = 2
# test_batch_size = 5
# number_of_epoch = 10
#
#
# start_time = datetime.datetime.now()
#
# # call methods to train our Convolutional Neural Network
# train_phase.train_phase(number_of_epoch, data_path)
#
# finish_train_time = datetime.datetime.now()
# duration_train = finish_train_time - start_time
# print('Training phase takes:', duration_train, '\n')
#
# # test our Convolutional Neural Network
# test_phase.test_phase(test_show_images_for_how_many_batch, test_batch_size, data_path)
#
# finish_test_time = datetime.datetime.now()
# duration_test = finish_test_time - finish_train_time
# print('Testing phase takes:', duration_test, '\n')

