import math
import numpy as np

def u(n):
    return 1 - n + n ** 2 - n ** 3 + n ** 4 - n ** 5 + n ** 6 - n ** 7 + n ** 8 - n ** 9 + n ** 10

uns = [u(n) for n in range(1,11)]

def lcm(a, b):
    return a * b // math.gcd(a, b)

assert lcm(1, -1) == -1

assert lcm(1, 1) == 1
assert lcm(3, 5) == 15
assert lcm(2, 4) == 4
assert lcm(4, 6) == 12


def solve_two_linear_equations(equation1, equation2):
    a1, b1, c1 = equation1
    a2, b2, c2 = equation2
    lcm_b = lcm(b1, b2)
    coef1 = lcm_b // b1
    coef2 = lcm_b // b2
    a1 *= coef1
    b1 *= coef1
    c1 *= coef1
    a2 *= coef2
    b2 *= coef2
    c2 *= coef2
    a = (c1 - c2) / (a1 - a2)
    b = (c1 - a1 * a) / b1
    return np.array([a, b])

assert solve_two_linear_equations((1,1,1), (2,1,683)).tolist() == [682, -681]
assert solve_two_linear_equations((2,3,8), (4,5,14)).tolist() == [1, 2]
assert solve_two_linear_equations((2,-3,2), (3,4,20)).tolist() == [4, 2]

def solve_linear_equations(equations):
    n = len(equations)
    if n == 2:
        return solve_two_linear_equations(equations[0], equations[1])
    else:
        new_equations = reduce_rows(equations)
        solution = solve_linear_equations(new_equations)
        solution = np.append(solution, [(equations[0][-1] - np.sum(equations[0][0:-2] * solution)) / equations[0][-2]])
        return solution

equation1 = np.array([1,1,1,1])
equation2 = np.array([4,2,1,683])
equation3 = np.array([9,3,1,838861])
equations = [equation1, equation2, equation3]
assert solve_linear_equations(equations).tolist() == [418748., -1255562.,   836815.]



def reduce_rows(equations):
    reduced_equations = []
    for i in range(len(equations) - 1):
        equation1 = equations[i]
        equation2 = equations[i+1]
        subtracted_equation = subtract_equations(equation1, equation2)
        reduced_equations.append(subtracted_equation)
    return reduced_equations


def subtract_equations(equation1, equation2):
    lcm_2nd_last = lcm(equation1[-2], equation2[-2])
    coef1 = lcm_2nd_last // equation1[-2]
    coef2 = lcm_2nd_last // equation2[-2]
    equation1 *= coef1
    equation2 *= coef2
    new_equation = equation2 - equation1
    return np.delete(new_equation, -2, 0)


assert subtract_equations(np.array([1,1,1,1]), np.array([4,2,1,683])).tolist() == [3, 1, 682]


def get_equations(n, uns):
    equations = []
    for i in range(n+1):
        equation = []
        for j in range(n+1):
            equation.append((i + 1) ** (n - j))
        equation.append(uns[i])
        equations.append(np.array(equation))
    return equations

assert get_equations(2, uns)[2].tolist() == [9,3,1, 44287]


def get_answer_at(solution, n):
    solution_len = len(solution)
    total = 0
    for i in range(solution_len):
        total += solution[i] * (n ** (solution_len - i - 1))
    return total


equations = get_equations(2, uns)
solution = solve_linear_equations(equations)
assert get_answer_at(solution, 1) == 1
assert get_answer_at(solution, 2) == 683
assert get_answer_at(solution, 3) == 44287
assert get_answer_at(solution, 4) == 130813


def get_fit(n, uns):
    equations = get_equations(n, uns)
    solution = solve_linear_equations(equations)
    return get_answer_at(solution, n+2)

assert get_fit(1, uns) == 1365


def u2(n):
    return n ** 3

u2ns = [u2(n) for n in range(1,10)]


assert get_fit(1, u2ns) == 15
assert get_fit(2, u2ns) == 58

for i in range(9):
    print(get_fit(i+1, uns))

sum([get_fit(i+1, uns) for i in range(9)]) + 1


