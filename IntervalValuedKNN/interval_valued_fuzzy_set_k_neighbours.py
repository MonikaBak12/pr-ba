from builtins import print

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.utils.multiclass import unique_labels
from tabulate import tabulate

from IntervalValuedKNN.cmp import CMP
from IntervalValuedKNN.iv_FS import IntervalValuedFuzzySet
from IntervalValuedKNN.iv_aggregations import AMean
from IntervalValuedKNN.iv_negations import Negation_1


class IntervalValuedFuzzyKNN(ClassifierMixin, BaseEstimator):
    def __init__(self, k_neighbours=3, weighted_similarity=False, decision_aggregation='similarities',
                 similarity_aggregation=AMean(), aggregation=AMean(), order='xu yager', precedence='w',
                 negation=Negation_1(), prec_aggregation=AMean()):
        self.k_neighbours = k_neighbours
        self.decision_aggregation = decision_aggregation
        self.weighted_similarity = weighted_similarity
        self.similarity_aggregation = similarity_aggregation
        self.aggregation = aggregation
        self.order = order
        self.prec_Agr = prec_aggregation
        self.negation = negation
        self.precedence = precedence
        self.chnging_dec_objs = []
        if precedence == 'w':
            self.precedence_function = self.precedence_w
        if precedence == 'z':
            self.precedence_function = self.precedence_z
        if precedence == 'a':
            self.precedence_function = self.precedence_a
        if precedence == 'Amean':
            self.precedence_function = self.prec_a_mean
        if precedence == 'AmeanPow':
            self.precedence_function = self.prec_a_meanPow
        if precedence == 'AmeanMax':
            self.precedence_function = self.prec_a_meanMax
        if precedence == 'p':
            self.precedence_function = self.prec_p
        if precedence == 'n':
            self.precedence_function = self.prec_n

    @staticmethod
    def change_nr_to_interval(val):
        x = 0.5 * np.minimum(val, 1 - val)
        lb = val * (1 - x)
        ub = lb + x
        return np.array([lb, ub])

    def r(self, x, y):
        return np.maximum(np.abs(x[0] - y[0]), np.abs(x[1] - y[1]))

    def w(self, x):
        return x[1] - x[0]

    def xu_yager_less_than(self, a, b):
        return a[0] + a[1] < b[0] + b[1] or (a[1] + a[0] == b[0] + b[1] and a[1] - a[0] <= b[1] - b[0])

    def l_order_less_than(self, a, b):
        return (a[0] < b[0] and a[1] <= b[1]) or (a[0] <= b[0] and a[1] < b[1])

    def fuzzy_and(self, a, b):
        return np.array([np.minimum(a[0], b[0], dtype=float), np.minimum(a[1], b[1], dtype=float)])

    def precedence_w(self, a, b):
        if IntervalValuedFuzzySet.partial_order_sharp(a, b):
            return np.ones(shape=(2,))
        return np.array([1.0 - np.maximum(self.w(a), self.r(a, b), dtype=float), 1.0 - self.r(a, b)])

    def inclusion_w2(self, a, b):
        np.apply_along_axis(self.xu_yager_less_than, arr=np.array(a, b))
        return np.array([1.0 - np.maximum(self.w(a), self.r(a, b), dtype=float), 1.0 - self.r(a, b)])

    def precedence_a(self, a, b):
        ord = IntervalValuedFuzzySet.xu_yager_less_than
        if self.order == 'partial':
            ord = IntervalValuedFuzzySet.partial_order
        if self.order == 'lex1':
            ord = IntervalValuedFuzzySet.lex_order_1
        if self.order == 'lex2':
            ord = IntervalValuedFuzzySet.lex_order_2
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        if ord(a, b):
            return np.array([1.0, 1.0])
        return self.aggregation.aggregate(np.array([1.0 - a, b]))

    def prec_a_mean(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        ord = IntervalValuedFuzzySet.partial_order
        if ord(a, b):
            return np.array([1.0, 1.0])
        agr = np.zeros((1, 2))
        agr[0] = np.array([(1.0 - a[1] + b[0]) / 2, (1.0 - a[0] + b[1]) / 2])
        return self.aggregation.aggregate(agr)

    def prec_a_meanPow(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        ord = IntervalValuedFuzzySet.partial_order
        if ord(a, b):
            return np.array([1.0, 1.0])
        agr = np.zeros((1, 2))
        agr[0] = np.array([(1.0 - a[1] + b[0]) / 2, np.sqrt(((1.0 - a[0]) * (1.0 - a[0]) + b[1] + b[1]) / 2)])
        return self.aggregation.aggregate(agr)

    def prec_a_meanMax(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        if IntervalValuedFuzzySet.partial_order(a, b):
            return np.array([1.0, 1.0])
        agr = np.zeros((1, 2))
        agr[0] = np.array([(1.0 - a[1] + b[0]) / 2.0, np.max([1.0 - a[0], b[1]])])
        return self.aggregation.aggregate(agr)

    """
    prec używany do Similarity "possible"
    """

    def prec_p(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        if IntervalValuedFuzzySet.possible_order(a, b):
            return np.array([1.0, 1.0])
        agr = np.zeros((1, 2))
        agr[0] = self.prec_Agr.aggregate(np.array([self.negation.negate_interval(a), b]))
        return agr[0]

    """
    prec używany do Similarity "necessary"
    """

    def prec_n(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        if IntervalValuedFuzzySet.necessary_order(a, b):
            return np.array([1.0, 1.0])
        return np.array([1.0 - np.max([self.w(a), self.r(a, b)]), 1.0 - self.r(a, b)])

    def precedence_z(self, a, b):
        ord = IntervalValuedFuzzySet.xu_yager_less_than
        if self.order == 'partial':
            ord = IntervalValuedFuzzySet.partial_order
        if self.order == 'lex1':
            ord = IntervalValuedFuzzySet.lex_order_1
        if self.order == 'lex2':
            ord = IntervalValuedFuzzySet.lex_order_2
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        if ord(a, b):
            return np.array([1.0, 1.0])
        return np.array([0.0, 0.0])

    def equivalence_measure_EAmeanZ(self, a, b):
        if np.all(a == b):
            return np.array([1 - self.w(a), 1])
        elif IntervalValuedFuzzySet.xu_yager_less_than_sharp(a, b) or IntervalValuedFuzzySet.xu_yager_less_than_sharp(b,
                                                                                                                      a):
            return np.array([0.5, 0.5])
        return np.array([0.0, 0.0])

    def indistinguishability_measure(self, a, b):
        a_aggregation = np.ndarray((self.X_.shape[1], 2))
        for i in range(self.X_.shape[1]):
            a_aggregation[i] = self.equivalence_measure_EAmeanZ(a[i], b[i])

        return self.similarity_aggregation.aggregate(a_aggregation)

    def similarity_measure(self, a, b, diff_num_attrib=None):
        num_attrib = a.shape[0]
        if diff_num_attrib is not None:
            num_attrib = diff_num_attrib
        beta_aggregation = np.ndarray((num_attrib, 2))
        for i in range(num_attrib):
            beta_aggregation[i] = self.fuzzy_and(self.precedence_function(a[i], b[i]),
                                                 self.precedence_function(b[i], a[i]))

        return self.similarity_aggregation.aggregate(beta_aggregation)

    def similarity_measure2(self, A, B):
        beta_aggregation = np.ndarray((self.X_.shape[1], 2))
        beta_aggregation = self.fuzzy_and(self.inclusion_w2(A, B), self.inclusion_w2(B, A))
        return AMean().aggregate(beta_aggregation)

    def make_interval_decision(self, fuzzy_decision):
        return np.array([fuzzy_decision * (1 - 0.5 * np.min(fuzzy_decision, 1 - fuzzy_decision)),
                         fuzzy_decision * (1 - 0.5 * np.min(fuzzy_decision, 1 - fuzzy_decision))
                         + 0.5 * np.min(fuzzy_decision, 1 - fuzzy_decision)])

    def feature_discretization(self, X):
        no_features = [2] * X.shape[1]
        return KBinsDiscretizer(n_bins=no_features, encode='ordinal', strategy='uniform').fit_transform(X)

    # def aggregate_decisions_by_abstraction_class(self, neighbours_indexes, neighbours_similarities):
    #     abstraction_classes = [[]]
    #     j = 0
    #     abstraction_classes[0].append(neighbours_indexes[-1])
    #     for i in range(neighbours_indexes.shape[0] - 1):
    #         print('neigbours similarities ', neighbours_similarities[neighbours_indexes])
    #         if np.array_equal(neighbours_similarities[neighbours_indexes[-i]],
    #                           neighbours_similarities[neighbours_indexes[-i - 1]]):
    #             abstraction_classes[j].append(-i - 1)
    #         else:
    #             abstraction_classes.append([])
    #             j += 1
    #             abstraction_classes[j].append(-i - 1)
    #     if len(abstraction_classes) == 1:
    #         neighbours_dec = self.y_[neighbours_indexes]
    #         negative_number = np.sum(neighbours_dec == 0)
    #         positive_number = self.k_neighbours - negative_number
    #         negative = negative_number / self.k_neighbours
    #         positive = positive_number / self.k_neighbours
    #         print('Interval one abstraction class', np.array([negative, negative]), np.array([positive, positive]))
    #         return self.assign_decision(np.array([negative, negative]), np.array([positive, positive]))
    #     lower_dec = self.y_[abstraction_classes[0]]
    #     lower_negative_number = np.sum(lower_dec == 0)
    #     lower_positive_number = len(lower_dec) - lower_negative_number
    #     upper_dec = self.y_[abstraction_classes[-1]]
    #     upper_negative_number = np.sum(upper_dec == 0)
    #     upper_positive_number = len(upper_dec) - upper_negative_number
    #     print('Interval ', np.array([lower_negative_number, upper_negative_number]),
    #           np.array([lower_positive_number, upper_positive_number]))
    #     return self.assign_decision(np.array([lower_negative_number, upper_negative_number]),
    #                                 np.array([lower_positive_number, upper_positive_number]))

    def assign_decision(self, positive_decision, negative_decision):

        ord = IntervalValuedFuzzySet.xu_yager_less_than
        if self.order == 'partial':
            ord = IntervalValuedFuzzySet.partial_order
        if self.order == 'lex1':
            ord = IntervalValuedFuzzySet.lex_order_1
        if self.order == 'lex2':
            ord = IntervalValuedFuzzySet.lex_order_2
        if self.order == 'possible':
            ord = IntervalValuedFuzzySet.possible_order
        if self.order == 'necessary':
            ord == IntervalValuedFuzzySet.necessary_order
        if self.w(positive_decision) < self.w(negative_decision):
            if positive_decision[0] >= 0.5:
                return 1
            elif negative_decision[0] >= 0.5:
                return 0
            elif positive_decision[1] >= 0.5:
                return 1
            elif negative_decision[1] >= 0.5:
                return 0
            else:
                return -1
        elif self.w(positive_decision) == self.w(negative_decision):
            if ord(positive_decision, negative_decision):
                if negative_decision[0] >= 0.5:
                    return 0
                elif positive_decision[0] >= 0.5:
                    return 1
                elif negative_decision[1] >= 0.5:
                    return 0
                elif positive_decision[1] >= 0.5:
                    return 1
                else:
                    return -1
            if ord(negative_decision, positive_decision):
                if positive_decision[0] >= 0.5:
                    return 1
                elif negative_decision[0] >= 0.5:
                    return 0
                elif positive_decision[1] >= 0.5:
                    return 1
                elif negative_decision[1] >= 0.5:
                    return 0
                else:
                    return -1
        elif negative_decision[0] >= 0.5:
            return 0
        elif positive_decision[0] >= 0.5:
            return 1
        elif negative_decision[1] >= 0.5:
            return 0
        elif positive_decision[1] >= 0.5:
            return 1
        else:
            return -1

    def distance(self, Syx, Syz):
        return np.amax([np.abs(Syx[0] - Syz[0]), np.abs(Syx[1] - Syz[1])])

    def aggregate_decs_by_similarities_or_ind(self, neighbours_indexes, neighbours_similarities):
        neighbours_dec = self.y_[neighbours_indexes]
        # print("DEC0= ", np.count_nonzero(self.y_ == 0))
        # print("DEC1= ", np.count_nonzero(self.y_ == 1))

        neighbours_with_negative_decision = neighbours_similarities[
            neighbours_indexes[np.argwhere(neighbours_dec == 0)]]
        neighbours_with_positive_decision = neighbours_similarities[
            neighbours_indexes[np.argwhere(neighbours_dec == 1)]]
        if not self.weighted_similarity:
            if len(neighbours_with_negative_decision) == 0:
                negative_decision = np.array([0.0, 0.0])
            else:
                negative_decision = self.aggregation.aggregate(fuzzy_sets=neighbours_with_negative_decision[0])
            if len(neighbours_with_positive_decision) == 0:
                positive_decision = np.array([0.0, 0.0])
            else:
                positive_decision = self.aggregation.aggregate(fuzzy_sets=neighbours_with_positive_decision[0])
        else:
            weights = np.arange(1.0, 0.0, -1 / self.k_neighbours)
            n = weights[np.argwhere(neighbours_dec == 0)]
            p = weights[np.argwhere(neighbours_dec == 1)]

            if len(neighbours_with_negative_decision) == 0:
                negative_decision = np.array([0.0, 0.0])
            else:
                for i in range(n.shape[0]):
                    k = np.multiply(n[i], neighbours_with_negative_decision[i][0].numpy_representation)
                    neighbours_with_negative_decision[i][0] = IntervalValuedFuzzySet.from_numpy(k, self.order)
                negative_decision = self.aggregation.aggregate(fuzzy_sets=neighbours_with_negative_decision[0])
            if len(neighbours_with_positive_decision) == 0:
                positive_decision = np.array([0.0, 0.0])
            else:
                for i in range(p.shape[0]):
                    k = np.multiply(p[i], neighbours_with_positive_decision[i][0].numpy_representation)
                    neighbours_with_positive_decision[i][0] = IntervalValuedFuzzySet.from_numpy(k, self.order)
                positive_decision = self.aggregation.aggregate(fuzzy_sets=neighbours_with_positive_decision[0])

        return self.assign_decision(positive_decision, negative_decision)

    def get_obj_nr_with_changing_dec(self):
        return self.chnging_dec_objs

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        indistinguishability = np.zeros((X.shape[0], self.X_.shape[0]), dtype=IntervalValuedFuzzySet)
        similarity = np.zeros((X.shape[0], self.X_.shape[0]), dtype=IntervalValuedFuzzySet)
        if self.decision_aggregation == 'indistinguishability':

            print(X.shape, "Indistinguishability", indistinguishability.shape)
            i = 0
            for test_object in X:
                j = 0
                for train_object in self.X_:
                    indistinguishability[i, j] = IntervalValuedFuzzySet.from_numpy(
                        self.indistinguishability_measure(test_object, train_object),
                        self.order)
                    j += 1
                i += 1
            sorted = np.argsort(indistinguishability)
            k_neighbours_indexes = sorted[:, -self.k_neighbours:]
            assigned_decisions = np.ndarray(shape=(X.shape[0],))
        else:
            print(X.shape, "similarity", similarity.shape)
            i = 0
            for test_object in X:
                j = 0
                for train_object in self.X_:
                    similarity[i, j] = IntervalValuedFuzzySet.from_numpy(
                        self.similarity_measure(test_object, train_object),
                        self.order)
                    j += 1
                i += 1
            sorted = np.argsort(similarity)
            self.sortedSimilarity = sorted
            self.sim = similarity

            k_neighbours_indexes = sorted[:, -self.k_neighbours:]
            assigned_decisions = np.ndarray(shape=(X.shape[0],))
        i = 0
        for k in k_neighbours_indexes:
            self.nr_test_obj = i
            if self.decision_aggregation == 'similarities':
                self.sim_k_0 = similarity[i][k]
                assigned_decisions[i] = self.aggregate_decs_by_similarities_or_ind(k, similarity[i])
            # if self.decision_aggregation == 'abstraction class':
            #     assigned_decisions[i] = self.aggregate_decisions_by_abstraction_class(k, similarity[i])
            if self.decision_aggregation == 'indistinguishability':
                assigned_decisions[i] = self.aggregate_decs_by_similarities_or_ind(k, indistinguishability[i])
            i += 1
        return assigned_decisions

    def get_similarities_for_obj(self, obj, miss_atrib):
        similarity = np.zeros((self.X_.shape[0]), dtype=IntervalValuedFuzzySet)
        j = 0
        obj2 = np.delete(obj, miss_atrib, 0)
        X_2 = self.X_
        self.X_ = np.delete(self.X_, miss_atrib, 1)
        for train_object in self.X_:
            similarity[j] = IntervalValuedFuzzySet.from_numpy(self.similarity_measure(obj2, train_object), self.order)
            j += 1
        self.X_ = X_2
        sorted = np.argsort(similarity)
        k_neighbours_indexes = sorted[-self.k_neighbours:]

        k_neighbours_obj = np.zeros((self.k_neighbours, 2), dtype=IntervalValuedFuzzySet)
        for k in range(self.k_neighbours):
            k_neighbours_obj[k] = self.X_[k_neighbours_indexes[k]][miss_atrib]

        return k_neighbours_obj

    def get_missing_attrib_by_entropy(self, obj, miss_atrib, meth=None):
        num_near_object = -1
        entropies = np.zeros(self.k_neighbours, dtype=IntervalValuedFuzzySet)
        similarity = np.zeros((self.X_.shape[0]), dtype=IntervalValuedFuzzySet)
        obj_without_miss_attrib = np.delete(obj, miss_atrib, 0)
        diffs = np.zeros(self.X_.shape[0])
        X_2 = self.X_
        self.X_ = np.delete(self.X_, miss_atrib, 1)
        for i, test_obj in enumerate(self.X_):
            similarity[i] = IntervalValuedFuzzySet.from_numpy(
                self.similarity_measure(obj_without_miss_attrib, test_obj), self.order)
            a = similarity[i].numpy_representation
            diffs[i] = a[1] - a[0]
        self.X_, X_2 = X_2, self.X_

        sorted = np.argsort(similarity)

        k_neighbours_indexes = sorted[-self.k_neighbours:]

        k_neighbours_obj = np.zeros((self.k_neighbours, X_2.shape[1], 2), dtype=IntervalValuedFuzzySet)
        negated_k_neighbours_obj = np.zeros((self.k_neighbours, X_2.shape[1], 2), dtype=IntervalValuedFuzzySet)
        for i, k in enumerate(k_neighbours_indexes):
            k_neighbours_obj[i] = X_2[k]
            if meth == "a":
                for j, n in enumerate(X_2[k]):
                    negated_k_neighbours_obj[i][j] = self.strong_negation(n)
            elif meth == "b":
                suma = 0
                for j, attrib in enumerate(X_2[k]):
                    if j != miss_atrib:
                        suma += attrib[1] - attrib[0]
                entropies[i] = suma
            elif meth == "c":
                suma = 0
                ones = np.ones((X_2.shape[1], 2), dtype=IntervalValuedFuzzySet)
                zeros = np.zeros((X_2.shape[1], 2), dtype=IntervalValuedFuzzySet)
                d0 = self.d(X_2[k], ones)
                d1 = self.d(X_2[k], zeros)
                if d1 > d0:
                    for e in X_2[k]:
                        if (1.0 - e[0]) > 0:
                            suma += e[1] / (1.0 - e[0])
                    entropies[i] = suma / X_2[k].shape[0]
                else:
                    for e in X_2[k]:
                        if e[1] > 0:
                            suma += (1.0 - e[0]) / e[1]
                    entropies[i] = suma / X_2[k].shape[0]

        if meth == "a":
            for i, obj in enumerate(k_neighbours_obj):
                entropies[i] = IntervalValuedFuzzySet.from_numpy(
                    self.similarity_measure(obj, negated_k_neighbours_obj[i],
                                            diff_num_attrib=k_neighbours_obj.shape[1]), order=self.order)

            array = entropies.copy()
            indexes = np.arange(self.k_neighbours)
            CMP.quick_sort(indexes, array, 0, len(array) - 1)
            num_near_object = indexes[0]

            the_same = 0
            for i in range(len(array) - 1):
                if CMP.equal(array[i].numpy_representation[0], array[i + 1].numpy_representation[0]) and \
                        CMP.equal(array[i].numpy_representation[0], array[i + 1].numpy_representation[0]):
                    the_same += 1
                else:
                    break
            if the_same > 0:
                print("takich samych", the_same + 1)
                print("indexy od największej do najmniejszej", indexes)
                best_sim = similarity[k_neighbours_indexes[indexes[0]]]
                best_obj_num = 0
                print(k_neighbours_indexes[indexes[0]])
                for i in range(1, the_same + 1):
                    if self.which_order(best_sim.numpy_representation,
                                        similarity[k_neighbours_indexes[indexes[i]]].numpy_representation):
                        best_obj_num = i
                        print("\t\tlepszy", best_sim, similarity[k_neighbours_indexes[indexes[i]]])
                    print(k_neighbours_indexes[indexes[i]])
                num_near_object = best_obj_num
        elif meth == "b" or meth == "c":
            sorted_entropy = np.sort(entropies)
            min = sorted_entropy[0]
            min_entr_indexes = np.where(entropies == min)
            count = np.count_nonzero(entropies == sorted_entropy[0])
            if count > 1:
                min_sim = np.zeros(count, dtype=IntervalValuedFuzzySet)
                for i, e in enumerate(min_entr_indexes):
                    min_sim[i] = similarity[k_neighbours_indexes[e[0]]]
                sort_min_sim = np.argsort(min_sim)
                num_near_object = sort_min_sim[0]
            else:
                num_near_object = min_entr_indexes[0][0]

        return self.X_[k_neighbours_indexes[num_near_object]][miss_atrib]

    def strong_negation(self, interval):
        return [1 - interval[1], 1 - interval[0]]

    def d(self, a, b):
        suma = 0
        for i, e in enumerate(a):
            suma += np.abs(e[0] - b[i][0]) + np.abs(e[1] - b[i][1]) + np.abs((e[1] - e[0]) - (b[i][1] - b[i][0]))
        return suma / a.shape[0]

    def which_order(self, a, b):
        if self.order == 'xu yager':
            return IntervalValuedFuzzySet.xu_yager_less_than(a, b)
        if self.order == 'partial':
            return IntervalValuedFuzzySet.partial_order(a, b)
        if self.order == 'lex1':
            return IntervalValuedFuzzySet.lex_order_1(a, b)
        if self.order == 'lex2':
            return IntervalValuedFuzzySet.lex_order_2(a, b)
        if self.order == 'possible':
            return IntervalValuedFuzzySet.possible_order
        if self.order == 'necessary':
            return IntervalValuedFuzzySet.necessary_order


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    print('DATA')
    print('ATRIBUTES')
    print(tabulate(X))
    print('DECISION')
    print(y)

    X = MinMaxScaler().fit_transform(X)
    discretized_X = np.zeros(shape=(X.shape[0], X.shape[1], 2))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            discretized_X[i, j] = IntervalValuedFuzzyKNN.change_nr_to_interval(X[i, j])
    print('INTERVAL')
    print(discretized_X.shape)
    X_train, X_test, y_train, y_test = train_test_split(discretized_X, y, test_size=0.3, random_state=10)

    knn = IntervalValuedFuzzyKNN(k_neighbours=3, weighted_similarity=False)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)
    print('CLASSIFIER ASSIGNED DECISIONS')
    print(predicted)
    print('y_test', y_test)
    print('ACC', knn.score(X_test, y_test))
