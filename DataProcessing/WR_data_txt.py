import numpy as np
def write_data(filename,data):
    with open(filename,'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i][0])+','+str(data[i][1])+','+str(data[i][2])+'\n')
def read_data(filename):
    data=[]
    with open(filename,'r') as file:
        for line in file.readline():
            data.append(float(i) for i in line.split(','))
    print(data)
def g_data():
    data=[]
    data=np.random.normal(size=(3,3))
    return data


if __name__ == '__main__':
    data=g_data()
    write_data("test.txt",data)
    read_data("test.txt")
