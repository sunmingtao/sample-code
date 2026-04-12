import math
import logging
import sys

#math.atan2()

class InputReader():

    triangles = []

    def __init__(self):
        self.LOGGER = logging.getLogger()
        text_file = open("p102.txt", "r")
        lines = text_file.read().split('\n')
        text_file.close()
        for line in lines:
            self.triangles.append(self.__parse_line(line))

    def __parse_line(self, line):
        xys = line.split(',')
        return [(int(xys[i*2]), int(xys[i*2+1])) for i in range(3)]

class Problem():
    def __init__(self, triangle):
        self.LOGGER = logging.getLogger()
        self.triangle = triangle

    def solve(self):
        angles = sorted([Problem.get_angle(a[0], a[1]) for a in self.triangle])
        self.__debug('Sorted angles are {}'.format(angles))
        return Problem.less_than_180_degree(angles[0], angles[1]) and Problem.less_than_180_degree(angles[1], angles[2]) and \
               Problem.less_than_180_degree(angles[2], angles[0] + 360)

    def get_angle(x, y):
        angle_in_radian = math.atan2(y, x)
        angle_in_degree = angle_in_radian * 180 / math.pi
        if angle_in_degree < 0:
            angle_in_degree += 360
        return angle_in_degree

    def less_than_180_degree(d1, d2):
        return d2 - d1 <= 180

    def __debug(self, message):
        self.LOGGER.debug(message)

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
    count = 0
    config_log(level=logging.DEBUG)
    inputReader = InputReader()
    triangles = inputReader.triangles
    for t in triangles:
        problem = Problem(t)
        if problem.solve():
            count += 1
    print(count)

if __name__ == "__main__":
    main()
