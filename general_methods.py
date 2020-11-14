from tabulate import tabulate

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

    precision_weighted_average = round((precision_NotAPerson*weight_NotAPerson+precision_Person*weight_Person+precision_PersonMask*weight_PersonMask)/3, 3)
    recall_weighted_average = round((recall_NotAPerson*weight_NotAPerson+recall_Person*weight_Person+recall_PersonMask*weight_PersonMask)/3, 3)
    f1measure_weighted_average = round((f1measure_NotAPerson*weight_NotAPerson+f1measure_Person*weight_Person+f1measure_PersonMask*weight_PersonMask)/3, 3)

    table_header = ['', 'precision', 'recall', 'f1-score', 'support']
    table_data = [
        ('NotAPerson', precision_NotAPerson, recall_NotAPerson, f1measure_NotAPerson, tp_NotAPerson+fn_NotAPerson),
        ('Person', precision_Person, recall_Person, f1measure_Person, tp_Person + fn_Person),
        ('PersonMask', precision_PersonMask, recall_PersonMask, f1measure_PersonMask, tp_PersonMask+fn_PersonMask),
        ('', '', '', ''),
        ('Average', precision_average, recall_average, f1measure_average),
        ('Weighted average', precision_weighted_average, recall_weighted_average, f1measure_weighted_average)
    ]
    print(tabulate(table_data, headers=table_header, tablefmt='grid', numalign="right", colalign=("left", "right", "right", "right")))