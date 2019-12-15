import numpy as np

filename = open('output_t4.txt', 'r')
output = open('t4.txt', 'w')
x = np.fromfile(filename, np.uint8)-np.float32(127.5)
l = x.tolist()
for i in range(0, len(l), 2):
    output.write(str(l[i]) + ' ' + str(l[i+1]) + '\n')

output.close()
