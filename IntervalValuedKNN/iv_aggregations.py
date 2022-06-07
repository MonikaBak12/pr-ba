import math

import numpy as np

from IntervalValuedKNN.iv_FS import IntervalValuedFuzzySet


class Aggregation(object):
    def __init__(self):
        pass

    def aggregate(self, fuzzy_sets):
        return self.aggregation_function(fuzzy_sets)

    def aggregation_function(self, fuzzy_sets):
        pass

    def aggregation_name(self):
        pass


class AMean(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return fs.sum(axis=0) / fs.shape[0]
        return fuzzy_sets.sum(axis=0) / fuzzy_sets.shape[0]

    def aggregation_name(self):
        return "A_Mean"


class AMeanPower(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.sum(fs[:, 0]) / fs.shape[0],
                             np.power(np.sum(np.square(fs[:, 1])) / fs.shape[0], 1.0 / fs.shape[0])])
        return np.array([np.sum(fuzzy_sets[:, 0]) / fuzzy_sets.shape[0],
                         np.power(np.sum(np.square(fuzzy_sets[:, 1])) / fuzzy_sets.shape[0],
                                  1.0 / fuzzy_sets.shape[0])])

    def aggregation_name(self):
        return "A_MeanPower"


class AMeanMax(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.sum(fs[:, 0]) / fs.shape[0], np.max(fs[:, 1])])
        return np.array([np.sum(fuzzy_sets[:, 0]) / fuzzy_sets.shape[0], np.max(fuzzy_sets[:, 1])])

    def aggregation_name(self):
        return "A_MeanMax"


class AProd(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return fs.prod(axis=0)
        return fuzzy_sets.prod(axis=0)

    def aggregation_name(self):
        return "A_Prod"


class AProdMean(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.prod(fs[:, 0]), np.sum(fs[:, 1]) / fs.shape[0]])
        return np.array([np.prod(fuzzy_sets[:, 0]), np.sum(fuzzy_sets[:, 1]) / fuzzy_sets.shape[0]])

    def aggregation_name(self):
        return "A_ProdMeam"


class AMinMax(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.min(fs[:, 0]), np.max(fs[:, 1])])
        return np.array([np.min(fuzzy_sets[:, 0]), np.max(fuzzy_sets[:, 1])])

    def aggregation_name(self):
        return "A_MinMax"


class AMin(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.min(fs[:, 0]), np.min(fs[:, 1])])
        return np.array([np.min(fuzzy_sets[:, 0]), np.min(fuzzy_sets[:, 1])])

    def aggregation_name(self):
        return "A_Min"


class AMax(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            return np.array([np.max(fs[:, 0]), np.max(fs[:, 1])])
        return np.array([np.max(fuzzy_sets[:, 0]), np.max(fuzzy_sets[:, 1])])

    def aggregation_name(self):
        return "A_Max"


class AAlpha(Aggregation):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            if fs.shape[0] > 1:
                agr = np.array([[self.alpha * fs[0, 0] + (1 - self.alpha) * fs[1, 0],
                                 self.alpha * fs[0, 1] + (1 - self.alpha) * fs[1, 1]]])
                for i in range(2, fs.shape[0]):
                    agr = np.array([[self.alpha * agr[0, 0] + (1 - self.alpha) * fs[i, 0],
                                     self.alpha * agr[0, 1] + (1 - self.alpha) * fs[i, 1]]])

            return agr[0]
        return np.array([self.alpha * fuzzy_sets[0, 0] + (1 - self.alpha) * fuzzy_sets[1, 0],
                         self.alpha * fuzzy_sets[0, 1] + (1 - self.alpha) * fuzzy_sets[1, 1]])

    def aggregation_name(self):
        return "A_Alpha_" + str(self.alpha)


class ALukasiewiczTnorm(Aggregation):
    @staticmethod
    def tl(x, y):
        return max(x + y - 1, 0)

    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            fs = np.array([f.numpy_representation for f in fuzzy_sets])
            agr = np.array([[self.tl(fs[0, 0], fs[1, 0]), self.tl(fs[0, 1], fs[1, 1])]])
            if fs.shape[0] > 2:
                for i in range(2, fs.shape[0]):
                    agr = np.array([[self.tl(agr[0, 0], fs[i, 0]), self.tl(agr[0, 1], fs[i, 1])]])
            return agr[0]
        return np.array([self.tl(fuzzy_sets[0, 0], fuzzy_sets[1, 0]), self.tl(fuzzy_sets[0, 1], fuzzy_sets[1, 1])])[0]

    def aggregation_name(self):
        return "A_LukasiewiczTnorm"


class A1(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            if len(fuzzy_sets) > 2:
                ones = IntervalValuedFuzzySet(1, 1)
                if fuzzy_sets[0] == ones and fuzzy_sets[1] == ones:
                    fuzzy_sets.__delitem__(0)
                    fuzzy_sets[0] = IntervalValuedFuzzySet.from_numpy(np.array([1, 1]))
                    return A1().aggregation_function(fuzzy_sets)
                else:
                    fs = np.array([f.numpy_representation for f in fuzzy_sets])
                    a = (fs[1][0] * (fs[0][0] + fs[0][1]) / 2) / 2
                    b = (fs[1][1] + fs[0][1]) / 2
                    fs = fs[1:]
                    fs[0] = [a, b]
                    fuzzy_sets = []
                    for x in fs:
                        fuzzy_sets.append(IntervalValuedFuzzySet.from_numpy(np.array(x)))
                    return A1().aggregation_function(fuzzy_sets)
            else:
                ones = IntervalValuedFuzzySet(1, 1)
                if fuzzy_sets[0] == ones and fuzzy_sets[1] == ones:
                    return np.array([1, 1])
                else:
                    fs = np.array([f.numpy_representation for f in fuzzy_sets])
                    a = (fs[1][0] * (fs[0][0] + fs[0][1]) / 2) / 2
                    b = (fs[1][1] + fs[0][1]) / 2
                    return np.array([a, b])
        return np.array([0, 0])

    def aggregation_name(self):
        return "A_1"


class A2(Aggregation):
    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            if len(fuzzy_sets) > 2:
                ones = IntervalValuedFuzzySet(1, 1)
                if fuzzy_sets[0] == ones and fuzzy_sets[1] == ones:
                    fuzzy_sets.__delitem__(0)
                    fuzzy_sets[0] = IntervalValuedFuzzySet.from_numpy(np.array([1, 1]))
                    return A2().aggregation_function(fuzzy_sets)
                else:
                    fs = np.array([f.numpy_representation for f in fuzzy_sets])
                    a = (fs[0][0] * (fs[1][0] + fs[1][1]) / 2) / 2
                    b = (fs[1][1] + fs[0][1]) / 2
                    fs = fs[1:]
                    fs[0] = [a, b]
                    fuzzy_sets = []
                    for x in fs:
                        fuzzy_sets.append(IntervalValuedFuzzySet.from_numpy(np.array(x)))
                    return A2().aggregation_function(fuzzy_sets)
            else:
                ones = IntervalValuedFuzzySet(1, 1)
                if fuzzy_sets[0] == ones and fuzzy_sets[1] == ones:
                    return np.array([1, 1])
                else:
                    fs = np.array([f.numpy_representation for f in fuzzy_sets])
                    a = (fs[0][0] * (fs[1][0] + fs[1][1]) / 2) / 2
                    b = (fs[1][1] + fs[0][1]) / 2
                    return np.array([a, b])
        return np.array([0, 0])

    def aggregation_name(self):
        return "A_2"


class A3(Aggregation):
    def __init__(self, agr='g'):
        # g-geometryczna, p- potegowa
        self.agr = agr

    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            ones = IntervalValuedFuzzySet(1, 1)
            if fuzzy_sets[0] == ones and fuzzy_sets[1] == ones:
                return np.array([1, 1])
            else:
                fs = np.array([f.numpy_representation for f in fuzzy_sets])
                b = 0
                if self.agr == 'g':
                    b = math.sqrt(fs[0][1] * fs[1][1])
                else:
                    b = (fs[0][1] + fs[1][1]) / 2
                return np.array([0, b])
        return np.array([0, 0])

    def aggregation_name(self):
        return "A_3"


class A4(Aggregation):
    def __init__(self, agr='g'):
        # g-geometryczna, p- potegowa
        self.agr = agr

    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            zeros = IntervalValuedFuzzySet(0, 0)
            if fuzzy_sets[0] == zeros and fuzzy_sets[1] == zeros:
                return np.array([0, 0])
            else:
                fs = np.array([f.numpy_representation for f in fuzzy_sets])
                b = 0
                if self.agr == 'g':
                    b = math.sqrt(fs[0][0] * fs[1][0])
                else:
                    b = (fs[0][0] + fs[1][0]) / 2
                return np.array([b, 1])

    def aggregation_name(self):
        return "A_4"


class AW(Aggregation):

    def aggregation_function(self, fuzzy_sets):
        if type(fuzzy_sets[0]) == IntervalValuedFuzzySet:
            if len(fuzzy_sets) < 2:
                return fuzzy_sets[0].numpy_representation
            else:
                fs = np.array([f.numpy_representation for f in fuzzy_sets])
                s1 = 0
                s2 = 0
                s3 = 0
                s4 = 0
                s5 = 0
                for i in range(fs.shape[0]):
                    s1 += fs[i][0]
                    s4 += fs[i][1]
                    s2 += fs[i][1] * (fs[i][1] - fs[i][0])
                    s5 += fs[i][0] * (fs[i][1] - fs[i][0])
                    s3 += fs[i][1] - fs[i][0]

                a = (s1 + s2) / (s3 + fs.shape[0])
                b = (s4 + s5) / (s3 + fs.shape[0])
                return np.array([a, b])
        return np.array([0, 0])

    def aggregation_name(self):
        return "A_W"


if __name__ == '__main__':
    x = IntervalValuedFuzzySet(0.5, 0.8)
    y = IntervalValuedFuzzySet(0.2, 0.6)
    z = IntervalValuedFuzzySet(0.4, 0.9)
    zz = IntervalValuedFuzzySet.from_numpy(np.array([0, 0.9]))

    two = [x, y]
    three = [x, y, z]
    ones = [IntervalValuedFuzzySet(1.0, 1.0), IntervalValuedFuzzySet(1.0, 1), IntervalValuedFuzzySet(0.1, 1)]

    print(AAlpha(0.5).aggregation_function(two))
    print(AAlpha(0.5).aggregation_function(three))
    print(ALukasiewiczTnorm().aggregation_function(two))
    print(ALukasiewiczTnorm().aggregation_function(three))
    print(AMin().aggregation_function(two))
    print(AMin().aggregation_function(three))
    print(AMax().aggregation_function(two))
    print(AMax().aggregation_function(three))
    print(AMinMax().aggregation_function(two))
    print(AMinMax().aggregation_function(three))
    print(AMean().aggregation_function(two))
    print(AMean().aggregation_function(three))
    print(AMeanMax().aggregation_function(two))
    print(AMeanMax().aggregation_function(three))
    print(AMeanPower().aggregation_function(two))
    print(AMeanPower().aggregation_function(three))
    print(AProdMean().aggregation_function(two))
    print(AProdMean().aggregation_function(three))
    print(AProd().aggregation_function(two))
    print(AProd().aggregation_function(three))
    print(AAlpha(0.5).aggregation_function(two))
    print(AAlpha(0.5).aggregation_function(three))
    print(two)
    print(three)
    print(A1().aggregation_function(two))
    print(A1().aggregation_function(three))
    print(A2().aggregation_function(two))
    print(A2().aggregation_function(three))
    print(A3().aggregation_function(two))
    print(A3().aggregation_function(three))
    print(A4().aggregation_function(two))
    print(A4().aggregation_function(three))
    print(A4().aggregation_function(ones))
    print((AW().aggregation_function([x])))
    print((AW().aggregation_function(two)))
    print((AW().aggregation_function(three)))
