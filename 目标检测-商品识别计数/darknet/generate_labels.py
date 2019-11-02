import csv
from PIL import Image

image_path = './data/'
out_path = './labels/'

train_csv = csv.reader(open('position.csv', 'r'))
header_row = next(train_csv)

for line in train_csv:
    image_name, class_num, x1, y1, x2, y2 = line
    img = Image.open(image_path + image_name)
    w, h = img.size

    x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)

    x = ((x1 + x2) / 2.0) / w
    y = ((y1 + y2) / 2.0) / h
    wid = (x2 - x1) / w
    hei = (y2 - y1) / h

    image_name = image_name.split('.')[0]
    out_file = open(out_path + image_name + '.txt', 'a')
    out_file.write(class_num + ' ' + str(x) + ' ' + str(y) + ' ' + str(wid) + ' ' + str(hei) + '\n')
    out_file.close()

print('Succeed!')

