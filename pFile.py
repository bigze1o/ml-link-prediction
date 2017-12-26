import sys

i = 0
data = []
i=0
for line in sys.stdin.readlines():
    get = line.split()
    if len(get) != 2:
        continue
    i += 1
    data.append(get[1])
    if i==3:
        print(','.join(data))
        i=0
        del data[:]
