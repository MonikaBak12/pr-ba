import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from sklearn.metrics import confusion_matrix

from IntervalValuedKNN.interval_valued_fuzzy_set_k_neighbours import IntervalValuedFuzzyKNN
from IntervalValuedKNN.iv_aggregations import AMean, AMeanPower, AMin, AMax, AMeanMax, A1, A2, A3, A4

y_train = []
entropy = False


def fill_missing_values(data_train, data_test, type, k_neighbours=3, meth=None, train=True):
    if type == 1:
        for w in range(data_test.shape[0]):
            for c in range(data_test.shape[1]):
                if np.isnan(data_test[w][c][0]):
                    data_test[w][c] = np.array([0, 1])
    else:
        marks_missing = np.zeros(shape=(data_test.shape[0], 2), dtype=int)
        data_filling_missings = data_test  # tablica danych z uzupełnionymi wartościami

        for w in range(data_test.shape[0]):  # oznaczanie obiektów z missingami
            for c in range(data_test.shape[1]):
                if np.isnan(data_test[w][c][0]):
                    marks_missing[w][0] = 1
                    marks_missing[w][1] = c

        for obj_num in range(marks_missing.shape[0]):
            if marks_missing[obj_num][0] == 1:
                miss_atrib = marks_missing[obj_num][1]

                knn = IntervalValuedFuzzyKNN(similarity_aggregation=s_agg, k_neighbours=k_neighbours, aggregation=aggg,
                                             order=ord, precedence=prec, weighted_similarity=False)
                if train:
                    new_X_train = data_filling_missings
                else:
                    new_X_train = data_train
                new_y = y_train
                test_obj = data_test[obj_num]
                if new_X_train.shape[0] != new_y.shape[0]:
                    new_y = np.append(new_y, new_y[new_y.shape[0] - 1])

                if train:  # jeśli dane są częścią treningową
                    test_obj_dec = new_y[obj_num]
                    for i in range(marks_missing.shape[0] - 1, -1, -1):
                        if type == 3 or entropy:
                            if marks_missing[i][0] == 1 or new_y[i] != test_obj_dec:
                                new_y = np.delete(new_y, i, 0)
                                new_X_train = np.delete(new_X_train, i, 0)
                        elif type == 2:
                            if marks_missing[i][0] == 1:
                                new_y = np.delete(new_y, i, 0)
                                new_X_train = np.delete(new_X_train, i, 0)

                knn.fit(new_X_train, new_y)

                if not entropy:
                    nears_attrib_value = knn.get_similarities_for_obj(test_obj, miss_atrib)
                    if nears_attrib_value.shape[0] < 5:

                        print(tabulate(nears_attrib_value))
                    if meth == "avg":
                        attrib_value = np.array(
                            [np.average(nears_attrib_value, axis=0)[0], np.average(nears_attrib_value, axis=0)[1]])
                    elif meth == "min":
                        attrib_value = np.array(
                            [np.min(nears_attrib_value, axis=0)[0], np.min(nears_attrib_value, axis=0)[1]])
                    elif meth == "max":
                        attrib_value = np.array(
                            [np.max(nears_attrib_value, axis=0)[0], np.max(nears_attrib_value, axis=0)[1]])
                    else:
                        attrib_value = np.array(
                            [np.min(nears_attrib_value, axis=0)[0], np.max(nears_attrib_value, axis=0)[1]])
                else:
                    attrib_value = knn.get_missing_attrib_by_entropy(test_obj, miss_atrib, meth)

                data_filling_missings[obj_num][miss_atrib] = attrib_value
                marks_missing[obj_num][0] = 0
        data_test = data_filling_missings
    return data_test


if __name__ == '__main__':
    f_mark = 'Entr'
    time = datetime.today()
    time2 = time
    today = datetime.today().strftime("%b-%d-%Y")
    path_prefix = '../data/tests/'
    missing = True
    entropy = True
    if missing:
        path_prefix += 'Missing/'

    data_slice = "[test0.3]"
    # dec_aggr = "indistinguishability"
    dec_aggr = 'similarity'
    # miss_rate = 0.05
    miss_rate = 0.2
    # miss_rate = 0.5
    type = 3
    meths = ('mm',)  # 'avg',
    # meths = ('[0,1]',)
    k_neighbours = (1,)
    # aggregations = (AMeanMax(), AMean())
    aggregations = (AMean(), ) # A1(), A2(), A3(), A4())
    # aggregations = # AMeanPower(), AMin(), AMax(),
    # aggregations = (AAlpha(0.25),)
    # aggregations = (AMeanMax(), AMean())
    # aggregations = (AMeanPower(),)
    # aggregations = (AMean(),)
    # aggregations = (AMax(),)
    # aggregations = (AMin(),)
    # similarity_aggregations = (AMean(), AMeanMax(), AMeanPower(),)
    similarity_aggregations = (A1(), A2(), A3(), A4())
    # precedences = ('Amean',)
    # precedences = ('AmeanMax',)
    precedences = ('p',)
    # precedences = ('n')
    # precedences = ('AmeanPow',)
    # precedences = ('w',) # 'w', 'a'
    # orders = ('lex2', 'partial', 'xu yager')  # ,'lex1', 'lex2')
    orders = ('possible',) # 'necessary')
    weighted_similarity = (False,)
    attempts = 1
    accuracy = []
    # np.ndarray((len(k_neighbours), len(aggregations), len(orders), len(precedences), len(weighted_similarity),
    #         attempts))
    sensitivity = []  # np.ndarray(accuracy.shape)
    specificity = []  # np.ndarray(accuracy.shape)
    precision = []  # np.ndarray(accuracy.shape)

    if entropy:
        type = "entropy"
        meths = ('c',)  # 'c'
    index = 0
    dfs = []
    for k in k_neighbours:
        for aggg in aggregations:
            for s_agg in similarity_aggregations:
                for ord in orders:
                    for meth in meths:
                        for prec in precedences:
                            accuracy.append([])
                            precision.append([])
                            specificity.append([])
                            sensitivity.append([])
                            for i in range(attempts):
                                if entropy and isinstance(s_agg, A1):
                                    prec = 'p'
                                # elif entropy and isinstance(s_agg, AMeanMax):
                                #    prec = 'AmeanMax'
                                # elif entropy and isinstance(s_agg, AMeanPower):
                                #    prec = 'AmeanPow'
                                # elif entropy and isinstance(s_agg, AMean):
                                #    prec = 'AmeanPow'
                                #elif entropy and isinstance(s_agg, A1):
                                #    prec = 'p'
                                elif entropy and isinstance(s_agg, A2):
                                    prec = 'p'
                                elif entropy and isinstance(s_agg, A3):
                                    prec = 'p'
                                elif entropy and isinstance(s_agg, A4):
                                    prec = 'p'

                                print('k=', k)
                                print('agg', aggg.aggregation_name())
                                print('similarity aggregation', s_agg.aggregation_name())
                                print('order', ord)
                                print('prec', prec)
                                print('attempt ', i)
                                print('method filling missings', miss_rate, type, meth)
                                print('decision agregation ', dec_aggr)
                                print('weighted ', False)
                                X_train = np.load(
                                    path_prefix + data_slice + 'X_train' + str(miss_rate) + str(i) + '.npy')
                                X_test = np.load(path_prefix + data_slice + 'X_test' + str(miss_rate) + str(i) + '.npy')
                                y_train = np.load(
                                    path_prefix + data_slice + 'y_train' + str(miss_rate) + str(i) + '.npy')
                                y_test = np.load(path_prefix + data_slice + 'y_test' + str(miss_rate) + str(i) + '.npy')

                                X_train = fill_missing_values(X_train, X_train, type=type, k_neighbours=k, meth=meth)
                                X_test = fill_missing_values(X_train, X_test, type=type, k_neighbours=k, meth=meth,
                                                             train=False)

                                knn = IntervalValuedFuzzyKNN(similarity_aggregation=s_agg, k_neighbours=k,
                                                             aggregation=aggg,
                                                             order=ord, precedence=prec,
                                                             weighted_similarity=False)
                               # print(y_test)
                               # print(predict)

                                knn.fit(X_train, y_train)
                                predict = knn.predict(X_test)
                                print("y_test", y_test.shape)
                                print("predict", predict.shape)
                                print(y_test)
                                print(predict)

                                cm = confusion_matrix(y_test, predict)
                                print('confusion matrix')
                                print(cm)
                                if cm.shape[0] == 2:
                                    tn, fp, fn, tp = cm.ravel()
                                else:
                                    t1, t2, t3, t4, tn, fp, t7, fn, tp = cm.ravel()
                                acc = (tp + tn) / (tp + tn + fp + fn)
                                sen = tp / (tp + fn)
                                spe = tn / (fp + tn)
                                pre = tp / (tp + fp)

                                accuracy[-1].append(acc)
                                sensitivity[-1].append(sen)
                                specificity[-1].append(spe)
                                precision[-1].append(pre)
                                print('acc = ', acc)
                                print('sensitivity=', sen)
                                print('specificity=', spe)
                                print('precision=', pre)
                            index += 1
                            print('acc = ')
                            print(accuracy)
                            print(accuracy[-1])
                            print(np.mean(accuracy[-1]))
                            print('sensitivity=')
                            print(sensitivity)
                            print(sensitivity[-1])
                            print(np.mean(sensitivity[-1]))
                            print('specificity=', spe)
                            print(specificity)
                            print(specificity[-1])
                            print(np.mean(specificity[-1]))
                            print('precision=', pre)
                            print(precision)
                            print(precision[-1])
                            print(np.mean(precision[-1]))

                            average = pd.DataFrame(
                                {'accuracy': np.mean(accuracy), 'sensitivity': np.mean(sensitivity),
                                 'specificity': np.mean(specificity), 'precision': np.mean(precision),
                                 'class aggregation': aggg.aggregation_name(),
                                 'similarity aggregation': s_agg.aggregation_name(), 'order': ord, 'precedence': prec,
                                 'k': k, 'method_fill_missings': meth},
                                index=[index])
                            dfs.append(average)
                            concatenated = pd.concat(dfs)
                            concatenated.to_excel(
                                path_prefix + 'result/part/' + f_mark + data_slice + 'part_Miss_Full' + str(
                                    miss_rate) + aggg.aggregation_name() + str(type) + today + meth + '.xlsx')
                            # + dec_aggr+ prec+ meth
                            print("\t\t\t\tCzas obliczeń", datetime.today() - time2)
                            time2 = datetime.today()
    concatenated = pd.concat(dfs)
    filename = path_prefix + 'result/' + f_mark + data_slice + '_Miss_Full' + str(
        miss_rate) + aggg.aggregation_name() + str(type) + today + meth + '.xlsx'
    concatenated.to_excel(filename)
    print(accuracy)
    print(precision)
    print(specificity)
    print(sensitivity)
    print()
    print(filename)
    print("czas całkowity", datetime.today() - time)