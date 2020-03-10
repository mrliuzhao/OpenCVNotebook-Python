import os


if __name__ == "__main__":
    # path = os.path.abspath('..')
    # path = os.path.join(path, 'test', 'abc.jpg')
    # print(path)

    for root, dirs, files in os.walk(r'.'):
        print('root:', root) #当前目录路径
        print('dirs:', dirs) #当前路径下所有子目录
        print('files:', files) #当前路径下所有非目录子文件





