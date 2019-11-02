import csv

train_file_path = './train/'
test_file_path = './test/'
train_csv = csv.reader(open('data.csv', 'r'))


if __name__ == "__main__":

    with open('train.txt', 'w') as f:
        for line in train_csv:
            img_path = train_file_path + line[0]
            label = line[1]
            f.write(img_path + ' ' + label + '\n')
                
    print('Succeed!')