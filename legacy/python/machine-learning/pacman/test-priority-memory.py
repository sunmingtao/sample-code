from memory import PrioritizedMemory
from util import SumSegmentTree

sum_tree = SumSegmentTree(capacity=8)

sum_tree[0] = 1
sum_tree[1] = 2
sum_tree[2] = 3
sum_tree[3] = 4
sum_tree[4] = 5
sum_tree[5] = 6
sum_tree[6] = 7
sum_tree[7] = 8

print(sum_tree.reduce(2,4))

