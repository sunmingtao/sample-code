import math
import numpy as np

def side_requires_cubes(a, b, layer):
    one = a * b
    two = (a * 2 + b * 2) * (layer - 1)
    three = (layer - 1) * (layer - 2) * 2
    return one, two, three

assert side_requires_cubes(3, 2, 1) == (6, 0, 0)
assert side_requires_cubes(3, 2, 2) == (6, 10, 0)
assert side_requires_cubes(3, 2, 3) == (6, 20, 4)

def require_cubes(a, b, c, layer):
    one1, two1, three1 = side_requires_cubes(a, b, layer)
    one2, two2, three2 = side_requires_cubes(b, c, layer)
    one3, two3, three3 = side_requires_cubes(a, c, layer)
    return (one1+one2+one3) * 2 + two1+two2+two3 + (three1+three2+three3)*2//3

assert require_cubes(3,2,1, 1) == 22
assert require_cubes(3,2,1, 2) == 46
assert require_cubes(3,2,1, 3) == 78
assert require_cubes(3,2,1, 4) == 118


def init_map(limit):
    cubu_map = {}
    a = 0
    while True:
        a += 1
        cubes = require_cubes(a, 1, 1, 1)
        if cubes > limit:
            break
        b = 0
        while b < a:
            b += 1
            cubes = require_cubes(a, b, 1, 1)
            if cubes > limit:
                break
            c = 0
            while c < b:
                c += 1
                cubes = require_cubes(a, b, c, 1)
                if cubes > limit:
                    break
                layer = 0
                while True:
                    layer += 1
                    cubes = require_cubes(a, b, c, layer)
                    if cubes > limit:
                        break
                    else:
                        if cubes in cubu_map:
                            cubu_map[cubes] += 1
                        else:
                            cubu_map[cubes] = 1
    return cubu_map


cube_map = init_map(200)

assert cube_map[22] == 2
assert cube_map[46] == 4
assert cube_map[78] == 5
assert cube_map[118] == 8
assert cube_map[154] == 10

def min_n(cube_num):
    cube_map = init_map(cube_num * 20)
    return min(a for a, b in cube_map.items() if b == cube_num)


assert min_n(10) == 154

print(min_n(1000))