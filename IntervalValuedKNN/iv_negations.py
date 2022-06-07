import numpy as np


class Negation(object):
    def __init__(self):
        pass

    def negate_interval(self, interval):
        pass

    def negation_name(self):
        pass

    def negation_formula(self):
        pass

"""
Strong negation
[1 - interval[1], 1 - interval[0]]
"""
class Negation_1(Negation):
    def negate_interval(self, interval):
        return [1 - interval[1], 1 - interval[0]]

    def negation_name(self):
        return "N_1 strong"


class Negation_2(Negation):
    def negate_interval(self, interval):
        if interval[0] == 0 and interval[1] == 0:
            return [1, 1]
        return [0, 1 - interval[1]]

    def negation_name(self):
        return "N_2"

class Negation_3(Negation):
    def negate_interval(self, interval):
        if interval[0] == 1 and interval[1] == 1:
            return [0, 0]
        return [1 - interval[0], 1]

    def negation_name(self):
        return "N_3"


class Negation_4(Negation):
    def negate_interval(self, interval):
        if interval[0] == 1 and interval[1] == 1:
            return [0, 0]
        if interval[0] == 0 and interval[1] == 0:
            return [1, 1]
        return [(1 - interval[0]) / 2, interval[1] / 2]

    def negation_name(self):
        return "N_4"


class Negation_5(Negation):
    def negate_interval(self, interval):
        if interval[0] == 1 and interval[1] == 1:
            return [0, 0]
        if interval[0] == 0 and interval[1] == 0:
            return [1, 1]
        return [0, interval[0]]

    def negation_name(self):
        return "N_5"


if __name__ == '__main__':
    a = np.array([0.2, 0.8])
    b = [0.2, 0.8]
    print(Negation_1().negate_interval(a))
    print(Negation_2().negate_interval(a))
    print(Negation_3().negate_interval(a))
    print(Negation_4().negate_interval(a))
    print(Negation_5().negate_interval(a))
    print(Negation_5().negate_interval(b))
