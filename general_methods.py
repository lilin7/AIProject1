from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# print accuracy, precision, recall, f1-score
def printTable(statistic_data):
    tp_NotAPerson = statistic_data[0][0]
    fp_NotAPerson = statistic_data[0][1]
    fn_NotAPerson = statistic_data[0][2]
    tp_Person = statistic_data[1][0]
    fp_Person = statistic_data[1][1]
    fn_Person = statistic_data[1][2]
    tp_PersonMask = statistic_data[2][0]
    fp_PersonMask = statistic_data[2][1]
    fn_PersonMask = statistic_data[2][2]

    precision_NotAPerson = round(tp_NotAPerson/ (tp_NotAPerson+fp_NotAPerson), 3) if (tp_NotAPerson+fp_NotAPerson!=0) else 0
    recall_NotAPerson = round(tp_NotAPerson/ (tp_NotAPerson+fn_NotAPerson), 3) if (tp_NotAPerson+fn_NotAPerson!=0) else 0
    f1measure_NotAPerson = round((2*precision_NotAPerson*recall_NotAPerson / (precision_NotAPerson+recall_NotAPerson)),3) if (precision_NotAPerson+recall_NotAPerson!=0) else 0

    precision_Person = round(tp_Person/ (tp_Person+fp_Person),3) if (tp_Person+fp_Person!=0) else 0
    recall_Person = round(tp_Person/ (tp_Person+fn_Person),3) if (tp_Person+fn_Person!=0) else 0
    f1measure_Person = round((2 * precision_Person * recall_Person / (precision_Person + recall_Person)), 3) if (precision_Person+recall_Person!=0) else 0

    precision_PersonMask = round(tp_PersonMask/ (tp_PersonMask+fp_PersonMask),3) if (tp_PersonMask+fp_PersonMask!=0) else 0
    recall_PersonMask = round(tp_PersonMask/ (tp_PersonMask+fn_PersonMask),3) if (tp_PersonMask+fn_PersonMask!=0) else 0
    f1measure_PersonMask= round((2 * precision_PersonMask * recall_PersonMask / (precision_PersonMask + recall_PersonMask) ), 3) if (precision_PersonMask + recall_PersonMask!=0) else 0

    precision_average = round((precision_NotAPerson+precision_Person+precision_PersonMask)/3, 3)
    recall_average = round((recall_NotAPerson+recall_Person+recall_PersonMask)/3, 3)
    f1measure_average = round((f1measure_NotAPerson+f1measure_Person+f1measure_PersonMask)/3, 3)

    total_number_of_testcase = tp_NotAPerson+fn_NotAPerson + tp_Person+fn_Person+tp_PersonMask+fn_PersonMask

    weight_NotAPerson = round((tp_NotAPerson+fn_NotAPerson) / total_number_of_testcase,3) if (total_number_of_testcase!=0) else 0
    weight_Person = round((tp_Person + fn_Person) / total_number_of_testcase,3) if (total_number_of_testcase!=0) else 0
    weight_PersonMask = round((tp_PersonMask+fn_PersonMask) / total_number_of_testcase,3) if (total_number_of_testcase!=0) else 0

    precision_weighted_average = round((precision_NotAPerson*weight_NotAPerson+precision_Person*weight_Person+precision_PersonMask*weight_PersonMask), 3)
    recall_weighted_average = round((recall_NotAPerson*weight_NotAPerson+recall_Person*weight_Person+recall_PersonMask*weight_PersonMask), 3)
    f1measure_weighted_average = round((f1measure_NotAPerson*weight_NotAPerson+f1measure_Person*weight_Person+f1measure_PersonMask*weight_PersonMask), 3)

    table_header = ['', 'precision', 'recall', 'f1-score', 'support']
    table_data = [
        ('NotAPerson', precision_NotAPerson, recall_NotAPerson, f1measure_NotAPerson),
        ('Person', precision_Person, recall_Person, f1measure_Person),
        ('PersonMask', precision_PersonMask, recall_PersonMask, f1measure_PersonMask),
        ('', '', '', ''),
        ('Average', precision_average, recall_average, f1measure_average),
        ('Weighted average', precision_weighted_average, recall_weighted_average, f1measure_weighted_average)
    ]
    print(tabulate(table_data, headers=table_header, tablefmt='grid', numalign="right", colalign=("left", "right", "right", "right")))

    # to print number of images in each class, uncomment this
    # table_header = ['', 'precision', 'recall', 'f1-score', 'support']
    # table_data = [
    #     ('NotAPerson', precision_NotAPerson, recall_NotAPerson, f1measure_NotAPerson, tp_NotAPerson+fn_NotAPerson),
    #     ('Person', precision_Person, recall_Person, f1measure_Person, tp_Person + fn_Person),
    #     ('PersonMask', precision_PersonMask, recall_PersonMask, f1measure_PersonMask, tp_PersonMask+fn_PersonMask),
    #     ('', '', '', ''),
    #     ('Average', precision_average, recall_average, f1measure_average),
    #     ('Weighted average', precision_weighted_average, recall_weighted_average, f1measure_weighted_average)
    # ]
    # print(tabulate(table_data, headers=table_header, tablefmt='grid', numalign="right", colalign=("left", "right", "right", "right")))


    return [[precision_NotAPerson, recall_NotAPerson, f1measure_NotAPerson, tp_NotAPerson+fn_NotAPerson],
            [precision_Person, recall_Person, f1measure_Person, tp_Person + fn_Person],
            [precision_PersonMask, recall_PersonMask, f1measure_PersonMask, tp_PersonMask+fn_PersonMask],
            [precision_average, recall_average, f1measure_average],
            [precision_weighted_average, recall_weighted_average, f1measure_weighted_average]]

# calculate the average accuracy, precision, recall, f1-score for the 10 iterations and print table
def average_measurements(table_list):
    number_of_iterations = len(table_list)

    iterations_table_header = ['', 'precision', 'recall', 'f1-score']
    iterations_table_data = []

    for table_index in range(number_of_iterations):
        table_str = str(number_of_iterations)+ '-Fold Cross Validation Iteration ' + str(table_index+1)
        # weighted_average data
        data_tuple_for_1_iteration = (table_str, table_list[table_index][4][0], table_list[table_index][4][1], table_list[table_index][4][2])
        iterations_table_data.append(data_tuple_for_1_iteration)

    print(tabulate(iterations_table_data, headers=iterations_table_header, tablefmt='grid', numalign="right",
                   colalign=("left", "right", "right", "right")))


    # for getting average:
    precision_NotAPerson_total, recall_NotAPerson_total, f1measure_NotAPerson_total, \
    precision_Person_total, recall_Person_total, f1measure_Person_total, \
    precision_PersonMask_total, recall_PersonMask_total, f1measure_PersonMask_total, \
    precision_average_total, recall_average_total, f1measure_average_total, \
    precision_weighted_average_total, recall_weighted_average_total, f1measure_weighted_average_total = 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,

    support_NotAPerson_total, support_Person_total, support_PersonMask_total = 0, 0, 0

    accuracy_total = 0.0000


    for table_index in range(number_of_iterations):
        precision_NotAPerson_total = precision_NotAPerson_total + table_list[table_index][0][0]
        recall_NotAPerson_total = recall_NotAPerson_total + table_list[table_index][0][1]
        f1measure_NotAPerson_total = f1measure_NotAPerson_total + table_list[table_index][0][2]
        support_NotAPerson_total = support_NotAPerson_total + table_list[table_index][0][3]

        precision_Person_total = precision_Person_total + table_list[table_index][1][0]
        recall_Person_total = recall_Person_total + table_list[table_index][1][1]
        f1measure_Person_total = f1measure_Person_total + table_list[table_index][1][2]
        support_Person_total = support_Person_total + table_list[table_index][1][3]

        precision_PersonMask_total = precision_PersonMask_total + table_list[table_index][2][0]
        recall_PersonMask_total = recall_PersonMask_total + table_list[table_index][2][1]
        f1measure_PersonMask_total = f1measure_PersonMask_total + table_list[table_index][2][2]
        support_PersonMask_total = support_PersonMask_total + table_list[table_index][2][3]

        precision_average_total = precision_average_total + table_list[table_index][3][0]
        recall_average_total = recall_average_total + table_list[table_index][3][1]
        f1measure_average_total = f1measure_average_total + table_list[table_index][3][2]

        precision_weighted_average_total = precision_weighted_average_total + table_list[table_index][4][0]
        recall_weighted_average_total = recall_weighted_average_total + table_list[table_index][4][1]
        f1measure_weighted_average_total = f1measure_weighted_average_total + table_list[table_index][4][2]

        accuracy_total = accuracy_total + table_list[table_index][5]


    print('\nAverage accuracy of the 10 iterations : %.2f %%' % ((accuracy_total/number_of_iterations) * 100))

    #table_header = ['', 'precision', 'recall', 'f1-score', 'support']
    table_header = ['', 'precision', 'recall', 'f1-score']
    table_data = [
        ('NotAPerson', round(precision_NotAPerson_total / number_of_iterations, 3),
         round(recall_NotAPerson_total / number_of_iterations, 3),
         round(f1measure_NotAPerson_total / number_of_iterations, 3),
         #round(support_NotAPerson_total / number_of_iterations, 0)
         ),
        ('Person', round(precision_Person_total / number_of_iterations, 3),
         round(recall_Person_total / number_of_iterations, 3), round(f1measure_Person_total / number_of_iterations, 3),
         #round(support_Person_total / number_of_iterations, 0)
         ),

        ('PersonMask', round(precision_PersonMask_total / number_of_iterations, 3),
         round(recall_PersonMask_total / number_of_iterations, 3),
         round(f1measure_PersonMask_total / number_of_iterations, 3),
         #round(support_PersonMask_total / number_of_iterations, 0)
         ),
        ('', '', '', ''),
        ('Average', round(precision_average_total / number_of_iterations, 3),
         round(recall_average_total / number_of_iterations, 3),
         round(f1measure_average_total / number_of_iterations, 3)),
        ('Weighted average', round(precision_weighted_average_total / number_of_iterations, 3),
         round(recall_weighted_average_total / number_of_iterations, 3),
         round(f1measure_weighted_average_total / number_of_iterations, 3))
    ]

    print(tabulate(table_data, headers=table_header, tablefmt='grid', numalign="right", colalign=("left", "right", "right", "right")))

# to show a confusion matrix
def show_confusion_matrix(conf_matrix):
    df_cm = pd.DataFrame(conf_matrix.numpy(),
                         index=['NotAPerson', 'Person', 'PersonMask'],
                         columns=['NotAPerson', 'Person', 'PersonMask'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="BuPu")
    plt.show()