train_file_path = './data/'
test_file_path = './test/'


if __name__ == "__main__":

    train = open('2019_test.txt', 'w')

    with open('file_name_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name = line.split()[0]
            img_path = train_file_path + file_name + '.jpg'
            train.write(img_path)
            train.write('\n')
    train.close()
                
    print('Succeed!')