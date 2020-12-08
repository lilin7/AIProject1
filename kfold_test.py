import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import general_methods



classes = ('NotAPerson', 'Person', 'PersonMask', ) # define 3 classes for training and testing datasets

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

#def imshow(img, labels, predicted):
def imshow(img, text):
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))  # change from （channels,imagesize,imagesize） to （imagesize,imagesize,channels)

    # # plt.imshow(np.uint8(npimg))
    # plt.imshow(npimg.astype(np.uint8))
    # plt.imshow(np.transpose(img.astype('uint8'), (1, 2, 0)))
    #plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))

    # unnormalize
    mn = img.min()
    mx = img.max()
    mx -= mn
    I = ((img - mn) / mx) * 255

    npimg = I.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8)) # change from （channels,imagesize,imagesize） to （imagesize,imagesize,channels)

    plt.text(0, 1.5, text, transform=plt.gca().transAxes)
    plt.show()


def kfold_test_phase(i_fold, test_loader, test_batch_size, show_images_for_how_many_batch, conf_matrix_total):
    print('\nBegin testing for K-fold Cross Validation iteration', i_fold)

    net = torch.load('./saved_net/net' + str(i_fold) + '.pkl')  # load our net parameters from file

    correct = 0  # number of correct prediction
    total = 0  # number of total test cases
    batch_counter = 0

    # Below is for measure the precision, recall and F1-measure
    # for class "NotAPerson" 0
    tp_NotAPerson, fp_NotAPerson, fn_NotAPerson = 0, 0, 0

    # for class "Person" 1
    tp_Person, fp_Person, fn_Person = 0, 0, 0

    # for class "PersonMask" 2
    tp_PersonMask, fp_PersonMask, fn_PersonMask = 0, 0, 0

    # for confusion matrix
    conf_matrix = torch.zeros(3, 3)

    show_image_count = 0

    for images, labels in test_loader:  # one batch
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_counter = batch_counter + 1

        if show_image_count < show_images_for_how_many_batch:
            # print for test reason
            print('\nOnly print', show_images_for_how_many_batch, 'batchs for demostration, can pass in parameter to modify how many batchs you want to demostrate:')
            print('\n*************For batch ' + str(batch_counter) + ' (' + str(
                test_batch_size) + ' images):*************')
            print('%-15s %-70s' % ("GroundTruth:",
                                   labels))  # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])
            print('%-15s %s' % ("Predicted:",
                                predicted))  # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])

            print('%-15s %s' % (
                'GroundTruth:', " ".join('%-12s' % classes[labels[number]] for number in range(labels.size(0)))))
            print('%-15s %s' % (
                'Predicted:', " ".join('%-12s' % classes[predicted[number]] for number in range(labels.size(0)))))

            text = 'GroundTruth:' + " ".join(
                '%-12s' % classes[labels[number]] for number in range(labels.size(0))) + '\nPredicted:    ' + " ".join(
                '%-12s' % classes[predicted[number]] for number in range(labels.size(0)))

            imshow(torchvision.utils.make_grid(images, nrow=5), text)

            show_image_count = show_image_count + 1

        # for confusion matrix
        conf_matrix = confusion_matrix(outputs, labels, conf_matrix)
        conf_matrix_total = confusion_matrix(outputs, labels, conf_matrix_total)

        for number in range(labels.size(0)):
            if classes[labels[number]] == "NotAPerson" and classes[predicted[number]] == "NotAPerson":
                tp_NotAPerson = tp_NotAPerson + 1
            elif classes[labels[number]] != "NotAPerson" and classes[predicted[number]] == "NotAPerson":
                fp_NotAPerson = fp_NotAPerson + 1
            elif classes[labels[number]] == "NotAPerson" and classes[predicted[number]] != "NotAPerson":
                fn_NotAPerson = fn_NotAPerson + 1

            if classes[labels[number]] == "Person" and classes[predicted[number]] == "Person":
                tp_Person = tp_Person + 1
            elif classes[labels[number]] != "Person" and classes[predicted[number]] == "Person":
                fp_Person = fp_Person + 1
            elif classes[labels[number]] == "Person" and classes[predicted[number]] != "Person":
                fn_Person = fn_Person + 1

            if classes[labels[number]] == "PersonMask" and classes[predicted[number]] == "PersonMask":
                tp_PersonMask = tp_PersonMask + 1
            elif classes[labels[number]] != "PersonMask" and classes[predicted[number]] == "PersonMask":
                fp_PersonMask = fp_PersonMask + 1
            elif classes[labels[number]] == "PersonMask" and classes[predicted[number]] != "PersonMask":
                fn_PersonMask = fn_PersonMask + 1

    if total != 0:
        accuracy = round(correct / total, 4)
        print('\nAccuracy of the K-fold Cross Validation test dataset : %.2f %%' % ((correct / total) * 100))
    else:
        accuracy = 0
        print('\nAccuracy of the K-fold Cross Validation test dataset : 0')

    # for printing precision, recall, f1measure
    table_for_one_iteration = general_methods.printTable([[tp_NotAPerson, fp_NotAPerson, fn_NotAPerson],
                                [tp_Person, fp_Person, fn_Person],
                                [tp_PersonMask, fp_PersonMask, fn_PersonMask]])

    table_for_one_iteration.append(accuracy)

    # for confusion matrix
    print('\nConfusion matrix for K-fold Cross Validation iteration', i_fold, ':')
    print(conf_matrix)

    general_methods.show_confusion_matrix(conf_matrix)

    print('\nFinished testing for K-fold Cross Validation iteration', i_fold)
    return table_for_one_iteration