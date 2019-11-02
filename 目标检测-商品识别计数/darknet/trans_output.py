import csv
import numpy as np

output = open('output.csv', 'a', newline='')
csv_write = csv.writer(output, dialect='excel')
csv_write.writerow(['ImageName', 'CategoryID'])

names = []
counts = []

with open('result1.txt', 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        if line.startswith('/'):
            words = line.split('/')
            img_name = words[9]
            img_name = img_name.split('\n')[0]
            names.append(img_name)
        else: 
            count = np.zeros(200)
            nums = line.split()
            for num in nums:
                count[int(num)-1] += 1
            count = count.tolist()
            counts.append(count)
print(len(names), len(counts))

for i in range(len(names)):
    row = []
    row.append(names[i])
    row.extend(counts[i])
    csv_write.writerow(row)

print("Succeed!")

