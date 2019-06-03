import os
from urllib.request import urlretrieve
from six.moves import urllib
def down_file(url,sasve_path):

    def report(a,b,c):
        """
        :param a: 已经下载的数据块数量
        :param b: 数据块的大小
        :param c: 远程数据块的大小
        :return: NONE
        """
        print("\rDownloading %.2f%%" %(a*b*100.0/c))
        #获取我们下载文件的名字,os.path.basename 为根文件的名字
    filename=os.path.basename(url)
    if not os.path.isfile(os.path.join(save_path,filename)):
        print("downloading from %s"% url)
        urlretrieve(url,os.path.join(save_path,filename),reporthook=report)
        print("\ndownload finished")
    else:
        print("File has exits")
    filesize=os.path.getsize(os.path.join(save_path,filename))
    print("file size is :%.2fMB"% (filesize/1024/1024))
    """
        urllib模块提供的urlretrieve()函数。urlretrieve()方法直接将远程数据下载到本地。
        *urlretrieve(url, filename=None, reporthook=None, data=None)

        参数filename指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
        参数reporthook是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，
        我们可以利用这个回调函数来显示当前的下载进度。
        参数data指post导服务器的数据，该方法返回一个包含两个元素的(filename, headers) 元组，
        filename 表示保存到本地的路径，header表示服务器的响应头
    """



if __name__ == '__main__':
    url="https://github.com/pandolia/qqbot.git"
    save_path="./"
    down_file(url,save_path)
