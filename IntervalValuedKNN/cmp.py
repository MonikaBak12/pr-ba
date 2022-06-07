import numpy as np


class CMP(object):
    # static variables
    THRESHOLD = 1000000
    THRESHOLD2 = 1.0 / THRESHOLD

    def __init__(self):
        pass

    def __init__(self, threshold):
        self.THRESHOLD = threshold

    @staticmethod
    def sum(a, b):
        return (a * CMP.THRESHOLD + b * CMP.THRESHOLD) / CMP.THRESHOLD

    @staticmethod
    def diff(a, b):
        return (a * CMP.THRESHOLD - b * CMP.THRESHOLD) / CMP.THRESHOLD

    @staticmethod
    def equal(a, b):
        return np.abs((a - b)) < CMP.THRESHOLD2

    @staticmethod
    def check(a, b, operation):
        result = {
            '<': lambda x, y: x < y,
            '>': lambda x, y: x > y,
            '<=': lambda x, y: x <= y,
            '>=': lambda x, y: x >= y,
            '==': lambda x, y: x == y
        }[operation](int(a * CMP.THRESHOLD), int(b * CMP.THRESHOLD))
        return result

    @staticmethod
    def partition(indexes, array, start, end):
        pivot = array[start]
        piv = indexes[start]
        low = start + 1
        high = end

        while True:
            while low <= high and array[high] < pivot:
                high = high - 1

            while low <= high and not array[low] < pivot:
                low = low + 1

            if low <= high:
                array[low], array[high] = array[high], array[low]
                indexes[low], indexes[high] = indexes[high], indexes[low]
            else:
                break

        array[start], array[high] = array[high], array[start]
        indexes[start], indexes[high] = indexes[high], indexes[start]

        return high

    @staticmethod
    def quick_sort(indexes, array, start, end):
        if start >= end:
            return

        p = CMP.partition(indexes, array, start, end)
        CMP.quick_sort(indexes, array, start, p - 1)
        CMP.quick_sort(indexes, array, p + 1, end)
