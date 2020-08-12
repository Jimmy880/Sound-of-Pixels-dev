import ffmpeg

import os
import glob
import argparse
import random
import fnmatch
import os.path as P
from tqdm import tqdm
import cv2 

def find_recursive(root_dir, ext=['.mp4']):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[-1] in ext:
                files.append(os.path.join(root, filename))
    return files

def generate(name, video_path, audio_path, frame_path):
    stream = ffmpeg.input(video_path, loglevel='error')
    if audio_path is not None:
        audio_stream = ( stream.audio
            # .filter("aresample", 16000)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='11025')
            .overwrite_output()
            .run()
        )


    if frame_path is not None:
        print(frame_path)
        video_stream = (
            stream
            .filter('fps', fps=8)
            .output(f'{frame_path}/%d.jpg')
            .overwrite_output()
            .run()
        )
        
        # probe = ffmpeg.probe(video_path)
        # video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # width = int(video_stream['width'])
        # height = int(video_stream['height'])
        # import numpy as np
        # out, _ = (
        #     stream
        #     .filter('fps', fps=8)
        #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        #     .run(capture_stdout=True)
        # )
        # video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        
        # return video
        # import h5py
        # with h5py.File(frame_path + '.hdf5', 'w', swmr=True) as f:
        #     f[name] = video
        # print(' Saved')
        # np.save(frame_path + '/main.npy', video)
        
        # from PIL import Image
        # Image.fromarray(video[0]).save(f'{frame_path}/cover.jpg')
        with open(f'{frame_path}/finish.txt', 'w') as f:
            f.write('finish')
        return None
    else:
        return None

def update_bar(*outputs):
    global bar
    bar.update()
    # tqdm.write(str(a))

import multiprocessing as MP
if __name__ == "__main__":
    prefix = 'data/solo'
    videos = find_recursive(f'{prefix}/video', ext=['.mp4', '.webm', '.mkv'])
    global bar
    bar = tqdm(desc='Generation', total = len(videos))

    pool = MP.Pool(12)
    res = dict()
    for video in videos:
        basename = P.splitext(P.basename(video))[0]
        instr = P.basename(P.dirname(video))
        # if instr not in ['Bassoon', 'Cello']:
        #     continue

        audio_dir = f'{prefix}/audio11k/{instr}'
        os.makedirs(audio_dir, exist_ok = True)
        audio_path = audio_dir + '/' + basename + '.wav'
        audio_path = None

        frame_dir = f'{prefix}/frames/{instr}/{basename}'
        os.makedirs(f'{prefix}/frames/{instr}/{basename}', exist_ok=True)
        # frame_dir = None
        if P.exists(f'{frame_dir}/finish.txt'):
            continue

        res[basename] = (frame_dir, pool.apply_async(generate, args=(basename, video, audio_path, frame_dir),callback=update_bar))
        
    pool.close()
    pool.join()
    for a, (dirname, v) in res.items():
        print(dirname)
        v.get()
    # import h5py
    # with h5py.File('./data/frames_2/main.hdf5', 'w', swmr=True) as f:
    #     for name, (filename, r) in tqdm(res.items()):
    #         r.get()
    #         f[name] = h5py.ExternalLink(P.abspath(filename), name)