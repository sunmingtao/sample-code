import logging
import sys

class NumProperty():

    def __init__(self):
        self.LOGGER = logging.getLogger()

    def bouncy_num_proportion(self, n):
        return sum([self.is_bouncy(a) for a in range(1, n+1)]) / n

    def is_bouncy(self, num):
        str_num = str(num)
        n = str_num[0]
        i = 1
        trend = None
        while i < len(str_num):
            new_trend = trend
            n_temp = str_num[i]
            if n_temp > n:
                new_trend = 'I'
            elif n_temp < n:
                new_trend = 'D'
            if self.is_change_trend(trend, new_trend):
                self.__debug('Found bouncy number: {}'.format(num))
                return True
            else:
                trend = new_trend
                n = n_temp
                i += 1
        return False

    def is_change_trend(self, old_trend, new_trend):
        if old_trend is None:
            return False
        return old_trend != new_trend

    def __debug(self, message):
        self.LOGGER.debug(message)

class Problem():
    numbers = []

    def __init__(self, limit):
        self.limit = limit
        for a in range(9):
            self.numbers.append('N')


    def solve(self):
        num_digit = 2
        while True:
            if num_digit == 2:
                start_index = 0
            else:
                start_index = num_digit - 2
            for i in numbers[10 ** start_index : 10 ** (num_digit - 1)]:
                for j in range(10):
                    pass


def config_log(level=logging.INFO,
               threshold=logging.WARNING,
               format="%(asctime)s %(filename)s [%(levelname)s] %(message)s",
               datefmt="%H:%M:%S"):
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(format, datefmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(logging.Formatter(format, datefmt))
    root.addHandler(stdout_handler)


def main():
    total = 0
    config_log(level=logging.INFO)
    num_property = NumProperty()
    bouncy_count = 0
    for n in range(1, 2178000):
        if num_property.is_bouncy(n):
            bouncy_count += 1
        if bouncy_count / n == 0.5:
            print (0.5, n)
        elif bouncy_count / n == 0.9:
            print(0.9, n)
        elif bouncy_count / n == 0.99:
            print (0.99, n)
            break


if __name__ == "__main__":
    main()


