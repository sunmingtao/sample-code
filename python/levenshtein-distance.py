import numpy as np

def replace_cost(source_char, target_char):
    return source_char.lower() != target_char.lower()


def calculate_distance(source, target):
    source_length = len(source)
    target_length = len(target)
    dp = np.zeros(shape=(source_length+1, target_length+1), dtype=np.int32)
    for i in range(source_length+1):
        for j in range(target_length+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                dp[i][j] = min(dp[i-1][j-1]+replace_cost(source[i-1], target[j-1]), dp[i-1][j]+1, dp[i][j-1]+1)
    return dp[source_length][target_length]

def calculate_accuracy(source, target):
    distance = calculate_distance(source, target)
    target_length = len(target)
    return (target_length - distance) / target_length


assert calculate_distance('sp', 'sp') == 0
assert calculate_distance('spot', 'spat') == 1
assert calculate_distance('abcde', 'e') == 4
assert calculate_distance('a', 'abcde') == 4
assert calculate_distance('a', 'bcdea') == 4
assert calculate_distance('weekend', 'week end') == 1


import pandas as pd

file_path = '/Users/msun/Documents/oral-history/oral-history.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
n_characters = 0
n_accurate_char_before = 0
n_accurate_char_after = 0
for index, data in df.iterrows():
    if index <= 2:
        before_correction = ' '.join(data[0].split())
        after_correction = ' '.join(data[1].split())
        ground_truths = ' '.join(data[2].split())
        ground_truths_length = len(ground_truths)
        n_characters += ground_truths_length
        before_distance = calculate_distance(before_correction, ground_truths)
        before_accurate_chars = ground_truths_length - before_distance
        before_accuracy = before_accurate_chars / ground_truths_length
        n_accurate_char_before += before_accurate_chars
        after_distance = calculate_distance(after_correction, ground_truths)
        after_accurate_chars = ground_truths_length - after_distance
        after_accuracy = after_accurate_chars / ground_truths_length
        n_accurate_char_after += after_accurate_chars
        print('{:.2f}%, {:.2f}%'.format(before_accuracy * 100, after_accuracy * 100))

total_before_accuracy = n_accurate_char_before / n_characters
total_after_accuracy = n_accurate_char_after / n_characters
print('Total before accuracy = {:.4f} Total after accuracy = {:.4f}'.format(total_before_accuracy, total_after_accuracy))


source1='''
J BOGLING I Waverley :Rinks to Visit Canberra		
Five links from the Waveiley Club will visit		
Canberra duiing the week end and will play the		
Canberia Club on the Tonest green at 2 pm, on		
Satuiday 4-11 members of the Club aie asked		  
t,o bring refreshments for atternoon tea "ty'lhere
will be ample room for piactlce jinks in addition
to those en- gaged against the visitors The		  
following have been chosen to repiesent Canboua		
- McKinsliy (skip), Mcinnes, Thoipe, Keith		  
Jones (skip), McParlanc, Spullln Gray Monahan		
(skip), Peuy, Elliott, Fleming irrancis (skip),		
Sonjoivl\lo, Gunn, and Rain \ Weatherstone		  
(?k,ip), Hoian jl^d waias aiyd^pe Ville __,		  
jWnteinmn (skip),' Deo, Whltpfoijd, U Wyloax		
t Mt
'''

source2='''
J BOGLING I Waverley Rinks to Visit Canberra Five links from the Waverley Club will visit Canberra during the week end and will play the Canberra Club on the Forrest green at 2 pm on Saturday 4-11 members of the Club are asked to bring refreshments for afternoon tea "there 
will be ample room for practice jinks in addition to those engaged against the visitors 
The
following have been chosen to represent Canberra
- McKinstry (skip), Mcinnes, Thorpe, Keith
Jones (skip), McFarlane, Spullln Gray Monahan
(skip), Percy, Elliott, Fleming Francis (skip),
Sonjoivl\lo, Gunn, and Rain A Weatherstone
(skip), Hoian old walls Raydene Ville in,
jWnteinmn (skip),' Deo, Whltpfoijd, U Wyloax t Mt
'''

target='''
BOWLING Waverley Rinks to Visit Canberra
Five rinks from the Waverley Club will visit Canberra during the weekend, and 
will play the Canberra Club on the Forrest green at 2 p.m., on Saturday.
All members of the Club are asked to bring refreshments for afternoon tea. 
There will be ample room for practice rinks in addition to those engaged against 
the visitors.
The following have been chosen to represent Canberra:
McKinstry (skip), McInnes, Thorpe, Keith.
Jones (skip), McFurlane, Scullin, Gray.
Monahan (skip), Perry, Elliott, Fleming.
Francis (skip), Somervillie, Gunn and Rain.
Weatherstone (skip), Horan, Edwards and Do Vile.
Waterman (skip), Dee, Whiteford, Wyles. 
'''


print(calculate_accuracy(source2, target))
print(calculate_accuracy(source1, target))

source3 = '''
J BOGLING I Waverley Rinks to Visit Canberra Five links from the Waverley Club will visit Canberra during the week end and will play the Canberra Club on the Forrest green at 2 pm on Saturday 4-11 members of the Club are asked to bring refreshments for afternoon tea "there 
will be ample room for practice jinks in addition to those engaged against the visitors The
following have been chosen to represent Canberra 
- McKinstry (skip), Mcinnes, Thorpe, Keith 
Jones (skip), McFarlane, Spullln Gray Monahan 
(skip), Percy, Elliott, Fleming Francis (skip), 
Sonjoivl\lo, Gunn, and Rain A Weatherstone 
(skip), Hoian old walls Raydene Ville in, 
jWnteinmn (skip),' Deo, Whltpfoijd, U Wyloax t Mt
'''

target3= '''
BOWLING Waverley Rinks to Visit Canberra Five rinks from the Waverley Club will visit Canberra during the weekend, and will play the Canberra Club on the Forrest green at 2 p.m., on Saturday. All members of the Club are asked to bring refreshments for afternoon tea. There will be ample room for practice rinks in addition to those engaged against 
the visitors. The following have been chosen to represent Canberra: 
McKinstry (skip), McInnes, Thorpe, Keith. 
Jones (skip), McFurlane, Scullin, Gray. 
Monahan (skip), Perry, Elliott, Fleming. 
Francis (skip), Somervillie, Gunn and Rain. 
Weatherstone (skip), Horan, Edwards and Do Vile. 
Waterman (skip), Dee, Whiteford, Wyles.
'''

target4='BOWLING Waverley Rinks to Visit Canberra Five rinks from the Waverley Club will visit Canberra during the weekend, and will play the Canberra Club on the Forrest green at 2 p.m., on Saturday. All members of the Club are asked to bring refreshments for afternoon tea. There will be ample room for practice rinks in addition to those engaged against  the visitors. The following have been chosen to represent Canberra: McKinstry (skip), McInnes, Thorpe, Keith. Jones (skip), McFurlane, Scullin, Gray. Monahan (skip), Perry, Elliott, Fleming. Francis (skip), Somervillie, Gunn and Rain. Weatherstone (skip), Horan, Edwards and Do Vile. Waterman (skip), Dee, Whiteford, Wyles.'
print(calculate_distance(source3, target3))

print(len(target4), len(target3))