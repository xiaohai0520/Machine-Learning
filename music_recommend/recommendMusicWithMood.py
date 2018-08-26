import surprise

import pickle


class Recommend():

    def __init__(self,emotion):
        self.sadlist = ['140975707']
        self.happylist = ['89258186']
        self.naturelist = ['111450065']
        self.amazinglist = ['77349670']
        self.playlistdic = pickle.load(open("playlist.pkl","rb"))

        self.songdic = pickle.load(open("song.pkl","rb"))
        self.algo = surprise.dump.load('./recommendation.model')[1]
        print(self.algo)
        self.emotion = emotion


    def getPlaylist(self):

        playlistid = ''
        if self.emotion == 'sad':
            playlistid = self.sadlist[0]
        elif self.emotion == 'happy':
            playlistid = self.happylist[0]
        elif self.emotion == 'nature':
            playlistid = self.naturelist[0]
        else:
            playlistid = self.amazinglist[0]


        playlistname = self.playlistdic[playlistid]
        #
        # print(playlistid + '   ' + playlistname + '++++++++++++++')

        playlist_inner_id = self.algo.trainset.to_inner_uid(playlistid)
        # print("内部id", playlist_inner_id)

        playlist_neighbors = self.algo.get_neighbors(playlist_inner_id, k=10)

        # print(playlist_neighbors)
        playlist_neighbors = list(self.algo.trainset.to_raw_uid(inner_id)
                              for inner_id in playlist_neighbors)

        # print('++++++++++++++++')
        #
        # print(playlist_neighbors)


        playlist_neighbors = list(self.playlistdic[playlist_id]
                              for playlist_id in playlist_neighbors)
        # print(playlist_neighbors)
        # print()
        # print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")

        index = 1
        print('根据心情为您推荐的十个歌单: \n')
        for playlist in playlist_neighbors:

            print(str(index) + '. ' + playlist)
            index += 1


# Recommend('sad').getPlaylist()

# current_playlist = list(name_id_dic.keys())[39]
# print(name_id_dic)
# print("歌单名称", current_playlist)

# 取出近邻
# 映射名字到id
# playlist_id = name_id_dic[current_playlist]
# print("歌单id", playlist_id)
# 取出来对应的内部user id => to_inner_uid
# playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
# print("内部id", playlist_inner_id)



# 把歌曲id转成歌曲名字
# to_raw_uid映射回去
# playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
#                        for inner_id in playlist_neighbors)
# playlist_neighbors = (id_name_dic[playlist_id]
#                        for playlist_id in playlist_neighbors)
#
# print()
# # print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
# print('根据心情为您推荐的十个歌单: \n')
# for playlist in playlist_neighbors:
#     print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))


