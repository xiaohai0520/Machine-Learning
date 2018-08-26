from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io

from surprise import KNNBaseline, Reader
from surprise import Dataset
import surprise

import pickle

# 重建歌单id到歌单名的映射字典

def savemodel():

    id_name_dic = pickle.load(open("playlist.pkl","rb"))
    print("加载歌单id到歌单名的映射字典完成...")
    # 重建歌单名到歌单id的映射字典
    name_id_dic = {}
    for playlist_id in id_name_dic:
        name_id_dic[id_name_dic[playlist_id]] = playlist_id
    print("加载歌单名到歌单id的映射字典完成...")


    file_path = os.path.expanduser('./163_music_suprise_format.txt')
    # 指定文件格式
    reader = Reader(line_format='user item rating', sep=',')
    # 从文件读取数据
    music_data = Dataset.load_from_file(file_path, reader=reader)
    # 计算歌曲和歌曲之间的相似度
    print("构建数据集...")
    trainset = music_data.build_full_trainset()
    #sim_options = {'name': 'pearson_baseline', 'user_based': False}

    print("开始训练模型...")
    #sim_options = {'user_based': False}
    #algo = KNNBaseline(sim_options=sim_options)
    algo = KNNBaseline()
    algo.fit(trainset)

    #储存训练模型
    print("---------储存模型")
    surprise.dump.dump('./recommendation.model', algo=algo)
    print('----over---')

    # algo = surprise.dump.load('./recommendation.model')[1]
    #
    # playlistid = '140975707'
    # playlistname = id_name_dic[playlistid]
    #
    # print(playlistid + '   ' + playlistname + '++++++++++++++')
    #
    # playlist_inner_id = algo.trainset.to_inner_uid(playlistid)
    # print("内部id", playlist_inner_id)
    #
    # playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)
    #
    # playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
    #                       for inner_id in playlist_neighbors)
    # playlist_neighbors = (id_name_dic[playlist_id]
    #                       for playlist_id in playlist_neighbors)
    # print()
    # # print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
    # print('根据心情为您推荐的十个歌单: \n')
    # for playlist in playlist_neighbors:
    #     print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))


savemodel()




