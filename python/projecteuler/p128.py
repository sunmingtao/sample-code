import math

def num_tiles_before_layer(layer):
    if layer == 1:
        return 0
    if layer == 2:
        return 1
    return 1 + 3 * (layer - 1) * (layer - 2)


assert num_tiles_before_layer(1) == 0
assert num_tiles_before_layer(2) == 1
assert num_tiles_before_layer(3) == 7
assert num_tiles_before_layer(4) == 19
assert num_tiles_before_layer(5) == 37


def tile(layer, index):
    if layer == 1:
        return 1
    else:
        max_tiles = (layer - 1) * 6
        if index >= max_tiles:
            raise ValueError('index is too big')
        else:
            num_tiles = num_tiles_before_layer(layer)
            return num_tiles + 1 + index


assert tile(1, 0) == 1
assert tile(2, 0) == 2
assert tile(2, 1) == 3
assert tile(2, 2) == 4
assert tile(2, 3) == 5
assert tile(2, 4) == 6
assert tile(2, 5) == 7
assert tile(3, 0) == 8
assert tile(4, 0) == 20
assert tile(4, 17) == 37
assert tile(5, 0) == 38


def good_index (index, layer):
    length = (layer - 1) * 6
    index = (length + index) % length
    return index

def init_tile_map(limit):
    tile_map = {}
    tile_map[(1,0)] = 1

    for layer in range(2, limit+1):
        if layer <= 3:
            for index in range(0, (layer - 1) * 6):
                tile_map[(layer, index)] = tile(layer, index)
        else:
            for index in range(-2, 2):
                index = good_index(index, layer)
                tile_map[(layer, index)] = tile(layer, index)
    return tile_map


def get_tile(tile_map, layer, index):
    if layer == 1:
        return tile_map[(1,0)]
    index = good_index(index, layer)
    return tile_map[(layer, index)]

tile_map = init_tile_map(4)
assert get_tile(tile_map, 1, 0) == 1
assert get_tile(tile_map, 2, 0) == 2
assert get_tile(tile_map, 2, 1) == 3
assert get_tile(tile_map, 2, -1) == 7
assert get_tile(tile_map, 2, 6) == 2
assert get_tile(tile_map, 4, 0) == 20
assert get_tile(tile_map, 4, 1) == 21
assert get_tile(tile_map, 4, -1) == 37
assert get_tile(tile_map, 4, -2) == 36


def init_neighbour_map(tile_map, limit):

    neighbour_map = {}
    neighbour_map[(1,0)] = [2,3,4,5,6,7]

    for layer in range(2, limit):
        for index in range (-1, 1):
            g_index = good_index(index, layer)
            neighbour_map[(layer, g_index)] = []
            neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer, g_index + 1))
            neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer, g_index - 1))
            if index == 0:
                neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer - 1, 0))
                neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -1))
                neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, 0))
                neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, 1))
            else:
                if layer == 2:
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer - 1, 0))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -1))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -2))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -3))
                else:
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer - 1, 0))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer - 1, -1))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -1))
                    neighbour_map[(layer, g_index)].append(get_tile(tile_map, layer + 1, -2))
    return neighbour_map


def is_prime(num):
    if num == 1:
        return False
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True



def pd(tile_map, neighbour_map, layer, index):
    tile = get_tile(tile_map, layer, index)
    return sum([is_prime(int(math.fabs(i-tile))) for i in neighbour_map[(layer, index)]])


def init_tile_list(limit):
    tile_map = init_tile_map(limit)
    neighbour_map = init_neighbour_map(tile_map, limit)
    tile_list = [1]
    for layer in range(2, limit):
        for index in range (-1, 1):
            index = good_index(index, layer)
            if (pd(tile_map, neighbour_map, layer, index)) == 3:
                tile_list.append(get_tile(tile_map, layer, index))
    return tile_list


tile_list = init_tile_list(200)
assert tile_list[9] == 271
assert tile_list[44] == 117019

tile_list = init_tile_list(100000)
print (tile_list[1999])