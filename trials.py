import os

imfiles = []

with open('submission.txt', 'r') as f:
    for line in sorted(f):
        imfiles.append(line.strip('\n')+'.png')

for f in imfiles:
    print(f)

