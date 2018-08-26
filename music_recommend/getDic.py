# coding: utf-8
import pickle
import sys


def parse_playlist_get_info(in_line, playlist_dic, playlistRe_dic, song_dic):
    contents = in_line.strip().split("\t")
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    playlist_dic[playlist_id] = name
    playlistRe_dic[name] = playlist_id
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_dic[song_id] = song_name + "\t" + artist
        except:
            print
            "song format error"
            print
            song + "\n"


def parse_file(in_file, out_playlist, out_playlistReverse, out_song):
    # 从歌单id到歌单名称的映射字典
    playlist_dic = {}
    # 从歌曲id到歌曲名称的映射字典
    song_dic = {}

    playlistRe_dic = {}

    for line in open(in_file,'r+', encoding="utf-8"):
        parse_playlist_get_info(line, playlist_dic, playlistRe_dic, song_dic)
    # 把映射字典保存在二进制文件中
    pickle.dump(playlist_dic, open(out_playlist, "wb"))

    pickle.dump(playlistRe_dic, open(out_playlistReverse, "wb"))

    # 可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pickle.dump(song_dic, open(out_song, "wb"))


parse_file("./163_music_playlist.txt", "playlist.pkl", "playlist_re.pkl","song.pkl")