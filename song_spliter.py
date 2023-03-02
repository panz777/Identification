import pandas as pd
import numpy as np
import os


def top_song_filter(all_player_data: pd.DataFrame, n):
    grouped_songs = list(all_player_data.groupby("song_id"))
    grouped_songs.sort(key=lambda a: len(a[1]), reverse=True)
    filtered_song = grouped_songs[n][1]
    return_data = []
    for player in filtered_song.itertuples():
        start = getattr(player, "start_idx")
        end = getattr(player, "end_idx")
        if start is None:
            continue
        frame_num = (end - start) / 23
        train_frame_num = int(frame_num * 0.7)
        val_frame_num = int(frame_num * 0.1)
        train_idx_range = (start, start + train_frame_num * 23)
        val_idx_range = (start + train_frame_num * 23, start + train_frame_num * 23 + val_frame_num * 23)
        test_idx_range = (start + train_frame_num * 23 + val_frame_num * 23, start + train_frame_num * 23 +
                          val_frame_num * 46)
        cluster_idx_range = (start + train_frame_num * 23 + val_frame_num * 46, end)
        return_data.append((train_idx_range, val_idx_range, test_idx_range, cluster_idx_range))
    print(len(return_data))
    return return_data

