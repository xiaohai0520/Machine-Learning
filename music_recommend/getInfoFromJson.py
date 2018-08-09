#coding: utf-8
import json
import sys

def parse_song_line(in_line):
    data = json.loads(in_line)
    name = data['result']['name']
    tags = ",".join(data['result']['tags'])
    subscribed_count = data['result']['subscribedCount']
    if (subscribed_count < 100):
        return False
    playlist_id = data['result']['id']
    song_info = ''
    songs = data['result']['tracks']
    for song in songs:
        try:
            song_info += "\t" + ":::".join(
                [str(song['id']), song['name'], song['artists'][0]['name'], str(song['popularity'])])
        except Exception:
            # print e
            print(song)
            continue
    return name + "##" + tags + "##" + str(playlist_id) + "##" + str(subscribed_count) + song_info


def parse_file(in_file, out_file):
    out = open(out_file, 'w', encoding='utf-8')
    count = 0
    for line in open(in_file,'r+', encoding="utf-8"):
        print(line)
        result = parse_song_line(line)

        if (result):
            print(result+ "\n")
            out.write(result.strip()+ "\n")
            count += 1
            if count == 100:
                break

    out.close()


parse_file("./playlist_detail_all.json", "./163_music_playlist.txt")