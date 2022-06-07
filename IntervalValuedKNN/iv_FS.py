import numpy as np

from IntervalValuedKNN.cmp import CMP


class IntervalValuedFuzzySet(object):
    def __init__(self, lower_bound, upper_bound, order='xu yager'):
        self.numpy_representation = np.array([lower_bound, upper_bound])
        self.order = order

    @staticmethod
    def from_numpy(numpy_representation, order='xu yager'):
        i = IntervalValuedFuzzySet(0, 0, order)
        i.numpy_representation = numpy_representation
        return i

    @staticmethod
    def xu_yager_less_than(a, b):
        return CMP.check(a[0] + a[1], b[0] + b[1], "<") or (
                CMP.check(a[1] + a[0], b[0] + b[1], "==") and CMP.check(a[1] - a[0], b[1] - b[0], "<="))

    @staticmethod
    def partial_order(a, b):
        return CMP.check(a[0], b[0], "<=") and CMP.check(a[1], b[1], "<=")

    @staticmethod
    def lex_order_1(a, b):
        return CMP.check(a[0], b[0], "<") or (CMP.check(a[0], b[0], "==") and CMP.check(a[1], b[1], "<="))

    @staticmethod
    def lex_order_2(a, b):
        return CMP.check(a[1], b[1], "<") or (CMP.check(a[1], b[1], "==") and CMP.check(a[0], b[0], "<="))

    @staticmethod
    def partial_order_sharp(a, b):
        return CMP.check(a[0], b[0], "<=") and CMP.check(a[1], b[1], "<=") and (
                CMP.check(a[0], b[0], "<") or CMP.check(a[1], b[1], "<"))

    @staticmethod
    def xu_yager_less_than_sharp(a, b):
        return CMP.check(a[0] + a[1], b[0] + b[1], "<") or (
                CMP.check(a[1] + a[0], b[0] + b[1], "==") and CMP.check(a[1] - a[0], b[1] - b[0], "<"))

    @staticmethod
    def lex_order_1_sharp(a, b):
        return CMP.check(a[0], b[0], "<") or (CMP.check(a[0], b[0], "==") and CMP.check(a[1], b[1], "<"))

    @staticmethod
    def lex_order_2_sharp(a, b):
        return CMP.check(a[1], b[1], "<") or (CMP.check(a[1], b[1], "==") and CMP.check(a[0], b[0], "<"))

    @staticmethod
    def possible_order(a, b):
        return CMP.check(a[0], b[1], "<")

    @staticmethod
    def necessary_order(a, b):
        return CMP.check(a[1], b[0], "<")

    def __lt__(self, other):  # <
        if self.order == 'xu yager':
            return self.xu_yager_less_than_sharp(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex1':
            return self.lex_order_1_sharp(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex2':
            return self.lex_order_2_sharp(self.numpy_representation, other.numpy_representation)
        if self.order == 'partial':
            return self.partial_order_sharp(self.numpy_representation, other.numpy_representation)
        if self.order == 'possible':
            return self.possible_order(self.numpy_representation, other.numpy_representation)
        if self.order == 'necessary':
            return self.necessary_order(self.numpy_representation, other.numpy_representation)

    def __le__(self, other):  # <=
        if self.order == 'xu yager':
            return self.xu_yager_less_than(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex1':
            return self.lex_order_1(self.numpy_representation, other.numpy_representation)
        if self.order == 'lex2':
            return self.lex_order_2(self.numpy_representation, other.numpy_representation)
        if self.order == 'partial':
            return self.partial_order(self.numpy_representation, other.numpy_representation)

    def __eq__(self, other):
        return CMP.check(self.numpy_representation[0], other.numpy_representation[0], "==") and CMP.check(
            self.numpy_representation[1], other.numpy_representation[1], "==")

    def __str__(self):
        return str(self.numpy_representation)

    def __repr__(self):
        return str(self.numpy_representation)

    def __truediv__(self, other):
        if type(other) == float:
            return self.numpy_representation / other

    def strong_negation(self):
        return np.array([1 - self.numpy_representation[1], 1 - self.numpy_representation[0]])


if __name__ == '__main__':
    a = IntervalValuedFuzzySet.from_numpy(np.array([0.2, 0.4]), order="lex2")
    b = IntervalValuedFuzzySet.from_numpy(np.array([0.1, 0.5]), order="lex2")
    print('a\t', a)
    print('b\t', b)
    print(a.order, 'wynik powinno być TRUE', a < b)
