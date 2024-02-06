# data
word = 'dearz' # 🦌

# Find all the ways you can split a word into 2 parts
# splits with a loop
splits_a = []
for i in range(len(word)+1):
    splits_a.append([word[:i],word[i:]])
for i in splits_a:
    print(i)
# same splits, done using a list comprehension
splits_b = [(word[:i], word[i:]) for i in range(len(word) + 1)]
for i in splits_b:
    print(i)

#Delete a letter from each string in the splits list.
splits = splits_a
deletes = []
print('word : ', word)
for L,R in splits:
    if R:
        print(L + R[1:], ' <-- delete ', R
# deletes with a list comprehension
splits = splits_a
deletes = [L + R[1:] for L, R in splits if R]
print(deletes)
print('*** which is the same as ***')
for i in deletes:
    print(i)

#Filter the created strings to only keep existing words              
vocab = ['dean','deer','dear','fries','and','coke']
edits = list(deletes)
print('vocab : ', vocab)
print('edits : ', edits)
candidates=[]
### START CODE HERE ###
candidates = set(deletes).intersection(vocab)
### END CODE HERE ###
print('candidate words : ', candidates)
