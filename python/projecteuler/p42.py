text_file = open("p42.txt", "r")
lines = text_file.read().replace('"', '').split(',')
text_file.close()

len(lines)

'SKY'

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def letter_value(letter):
    return letters.find(letter)+1

def word_value(word):
    return sum(letter_value(w) for w in word)

word_value('SKY')


triangle_numbers = [triangle(n) for n in range(1, 21)]

for wd in lines:
    print (word_value(wd))

sum(word_value(wd) in triangle_numbers for wd in lines)

max(word_value(w) for w in lines)

192

n(n+1)/2
n = 40

def triangle(n):
    return int(n * (n + 1) / 2)

triangle(20)

