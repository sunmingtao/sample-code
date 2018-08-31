import numpy as np
import re

'''Convert 'ABC' to [[1,0,0],[0,1,0],[0,0,1]] '''
def string_to_one_hot_vector(string, known_chars='ABCDEFG'):
    value_to_index_dict = {value: index for index, value in enumerate(known_chars)} # {'A':0, 'B':1, 'C':2}
    output = np.zeros((len(string), len(known_chars)), dtype=np.int32)
    for i, value in enumerate(string):
        output[i, value_to_index_dict[value]] = 1
    return output


'''Randomly replace a char in a string. e.g. 'ABCDE' -> 'ABBDE' '''
def replace_char_in_string(string):
    index = np.random.randint(len(string))
    old_char = string[index]
    new_char_list = sorted(set(string) - set(old_char)) # sorted list
    new_char = np.random.choice(new_char_list)
    return string[:index]+ new_char + string[index+1:]

'''Replace 
Example: replace_regex_in_string('{abc}', r'[{}]', '') 
'''
def replace_regex_in_string(string, regex, to_replace):
    return re.sub(regex, to_replace, string)







