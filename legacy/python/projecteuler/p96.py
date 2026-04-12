import fractions
import logging
import math
import sys
import numpy as np

class Problem():

    def __init__(self):
        self.LOGGER = logging.getLogger()
        self.all_games = self.__read_games()
        self.benchmark()

    def __read_game(self, game):
        board = []
        for row in game[1:]:
            board_row = []
            for col in row:
                board_row.append(int(col))
            board.append(board_row)
        return board

    def __read_file(self):
        text_file = open("p96.txt", "r")
        lines = text_file.read().split('\n')
        text_file.close()
        return lines


    def __read_games(self):
        all_games = []
        lines = self.__read_file()
        for i in range(0, len(lines) // 10):
            all_games.append(self.__read_game(lines[i*10:i*10+10]))
        return all_games

    def benchmark(self):
        self.__assert(len(self.all_games), 50)

    def __assert(self, actual, expected, epsilon=1e-5):
        assert (abs(actual - expected) < epsilon)

class Sudoku():

    def __init__(self, game):
        self.LOGGER = logging.getLogger()
        self.game = np.array(game)
        self.__debug('Before solve:\n {}'.format(self.game))

    def solve(self):
        self.repeat_fill_no_guess_answer()
        if self.found_all_answers():
            self.__debug('Found the solution!')
            return self.game
        elif self.is_dead_end():
            self.__debug('Game reached a dead end, try new guess')
            return None
        else:
            possible_values, i, j = self.find_first_multiple_possible_values()
            self.__debug('There is no direct answer. Possible values at grid [{}][{}] are {}'.format( i, j, possible_values))
            for p in possible_values:
                self.__debug('Take a guess {} at grid [{}][{}]'.format(p, i, j))
                guess_game = self.game.copy()
                guess_game[i][j] = p
                guess_sudoku = Sudoku(guess_game)
                guess_game_result = guess_sudoku.solve()
                if guess_game_result is not None:
                    return guess_game_result



    def repeat_fill_no_guess_answer(self):
        found = self.fill_no_guess_answer()
        while found:
            found = self.fill_no_guess_answer()

    def fill_no_guess_answer(self):
        found = False
        for i in range(9):
            for j in range(9):
                if self.game[i][j] == 0:
                    possible_values = self.possible_values(i, j)
                    if len(possible_values) == 1:
                        self.__debug('Found answer {} to grid [{}][{}]'.format(list(possible_values)[0], i,j))
                        self.game[i][j] = list(possible_values)[0]
                        found = True
        return found

    def is_dead_end(self):
        for i in range(9):
            for j in range(9):
                if self.game[i][j] == 0:
                    possible_values = self.possible_values(i, j)
                    if len(possible_values) == 0:
                        self.__debug('Dead end for grid [{}][{}]'.format(i, j))
                        return True
        return False

    def found_all_answers(self):
        for i in range(9):
            for j in range(9):
                if self.game[i][j] == 0:
                    return False
        return True

    def find_first_multiple_possible_values(self):
        for i in range(9):
            for j in range(9):
                if self.game[i][j] == 0:
                    possible_values = self.possible_values(i, j)
                    if len(possible_values) > 1:
                        return list(possible_values), i, j
        return None


    def possible_values(self, i, j):
        all_row_values = self.get_all_row_values(i)
        all_col_values = self.get_all_column_values(j)
        all_grid_values = self.get_all_grid_values(i, j)
        return set(range(1,10)) - set(all_row_values) - set(all_col_values) - set(all_grid_values)

    def get_all_row_values(self, i):
        return [a for a in self.game[i] if a > 0]

    def get_all_column_values(self, j):
        return [a for a in self.game[:, j] if a > 0]

    def get_all_grid_values(self, i, j):
        i_range, j_range = i // 3, j // 3
        return [a for a in self.game[i_range*3:i_range*3+3, j_range*3:j_range*3+3].reshape(-1) if a > 0]

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
    total = 0
    config_log(level=logging.INFO)
    problem = Problem()
    for game in problem.all_games:
        sudoku = Sudoku(game)
        game_result = sudoku.solve()
        top_left = int(''.join([str(a) for a in game_result[0][0:3]]))
        print(top_left)
        total+=top_left
    print(total)
    #print(sudoku.get_all_row_values(1))
    #print(sudoku.get_all_column_values(6))
    #print(sudoku.get_all_grid_values(1,6))
    #print(sudoku.possible_values(1,6))
    #print(sudoku.solve())


if __name__ == "__main__":
    main()
