import datetime
import train_phase
import test_phase


start_time = datetime.datetime.now()

# call methods to train our Convolutional Neural Network
train_phase.train_phase()

finish_train_time = datetime.datetime.now()
duration_train = finish_train_time - start_time
print('Training phase takes:', duration_train, '\n')

# test our Convolutional Neural Network
test_phase.test_phase()

finish_test_time = datetime.datetime.now()
duration_test = finish_test_time - finish_train_time
print('Testing phase takes:', duration_test, '\n')

