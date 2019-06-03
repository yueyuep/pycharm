#!/usr/bin/python
# -*- coding:utf-8 -*-
import requests
import json
# import pymongo
import time


def main(name, filename):
    albumname = []
    content = []
    item = []
    for i in range(20):
        print("=======================爬取结果==========================\n")
        # 爬取第i页的结果
        print(i)
        # qq音乐的api
        url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'
        # 这里数据只有三个是需要变的，分别是：jsonpCallback，w(我们的查找的歌手名字) searchid
        data = {'qqmusic_ver': 1298,
                'remoteplace': 'txt.yqq.lyric',
                'inCharset': 'utf8',
                'sem': 1, 'ct': 24, 'catZhida': 1, 'p': i,
                'needNewCode': 0, 'platform': 'yqq',
                'lossless': 0, 'notice': 0, 'format': 'jsonp', 'outCharset': 'utf-8', 'loginUin': 0,
                'jsonpCallback': 'MusicJsonCallback19507963135827455',
                'searchid': '98485846416392878',
                'hostUin': 0, 'n': 10, 'g_tk': 5381, 't': 7,
                'w': name, 'aggr': 0
                }
        # 模仿浏览器进行请求
        headers = {'content-type': 'application/json',
                   'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
        r = requests.get(url, params=data, headers=headers)
        # print(r.text)
        time.sleep(3)
        text = r.text[35:-1]

        # 截取 第35个字符到最后一个
        result = json.loads(text)
        if result['code'] == 0:

            for list in result['data']['lyric']['list']:
                # print(list)
                albumname.append(list["albumname"])
                content.append(list["content"])
                print("专辑：{}\n内容为：{}\n".format(list["albumname"], list["content"]))

    # 将爬取的信息进行保存
    with open(filename, "w+") as file:
        for i in range(len(content)):
            mstr = "专辑名字\t" + str(albumname[i]) + "内容\t" + str(content[i])
            file.write(mstr)
            file.write("\n\n")


if __name__ == '__main__':
    # 页数最多为20 ，根据歌手的歌曲多少决定
    name = "周杰伦"
    filename = name + "歌单.txt"
    main(name, filename)

