# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import argparse
import os
from pathlib import Path
import time
import numpy as np
import scipy.ndimage

import pickle
import training.misc as misc
import dnnlib.tflib.tfutil as tfutil

#----------------------------------------------------------------------------
# main function for command line call

def main():

    starttime = int(time.time())
    parser = argparse.ArgumentParser(
        description='Render from StyleGAN2 saved models (pkl files).',
        #epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--network_pkl', help='The pkl file to render from (the model checkpoint).', default=None, metavar='MODEL.pkl', required=True)
    parser.add_argument('--grid_x', help='Number of images to render horizontally (each frame will have rows of X images, default: 1).', default=1, metavar='X', type=int)
    parser.add_argument('--grid_y', help='Number of images to render vertically (each frame will have cols of Y images, default: 1).', default=1, metavar='Y', type=int)
    parser.add_argument('--png_sequence', help='If True, outputs a folder of frames as pngs instead of video. (default: False).', default=False, type=bool)
    parser.add_argument('--image_shrink', help='Render in 1/[image_shrink] resolution (fast, useful for quick previews)', default=1, type=int)
    parser.add_argument('--image_zoom', help='Zoom on the output image (seems like just more video pixels, but no true upscaling)', default=1, type=float)
    parser.add_argument('--duration_sec', help='Length of video to render in seconds.', default=30.0, type=float)
    parser.add_argument('--mp4_fps', help='Frames per second for video rendering', default=30, type=float)
    parser.add_argument('--smoothing_sec', help='Gaussian kernel size in seconds to blend video frames (higher value = less change, lower value = more erratic, default: 1.0)', default=1.0, type=float)
    parser.add_argument('--truncation_psi', help='Truncation parameter (1 = normal, lower values overfit to look more like originals, higher values underfit to be more abstract, recommendation: 0.5-2)', default=1, type=float)
    parser.add_argument('--randomize_noise', help='If True, adds noise to vary rendered images.', default=False, type=bool)
    parser.add_argument('--filename', help='Filename for rendering output, defaults to pkl filename', default=None)
    parser.add_argument('--mp4_codec', help='Video codec to use with moviepy (i.e. libx264, libx265, mpeg4)', default='libx265')
    parser.add_argument('--mp4_bitrate', help='Bitrate to use with moviepy (i.e. 16M)', default='16M')
    parser.add_argument('--random_seed', help='Seed to initialize the latent generation.', default=starttime, type=int)
    parser.add_argument('--minibatch_size', help='Size of batch rendering (doesn\'t seem to have effects but left in anyway)', default=8, type=int)

    args = parser.parse_args()

    tfutil.init_tf()

    generate_interpolation_video(
        network_pkl=args.network_pkl, 
        grid_size=[args.grid_x, args.grid_y], 
        png_sequence=args.png_sequence, 
        image_shrink=args.image_shrink, 
        image_zoom=args.image_zoom, 
        duration_sec=args.duration_sec, 
        smoothing_sec=args.smoothing_sec, 
        truncation_psi=args.truncation_psi,
        randomize_noise=args.randomize_noise,
        filename=args.filename, 
        mp4_fps=args.mp4_fps, 
        mp4_codec=args.mp4_codec, 
        mp4_bitrate=args.mp4_bitrate,
        random_seed=args.random_seed, 
        minibatch_size=args.minibatch_size
    )

#----------------------------------------------------------------------------
# Helper functions that have been dropped from pgan to sgan

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

def load_pkl(network_pkl):
    with open(network_pkl, 'rb') as file:
        return pickle.load(file, encoding='latin1')

def get_id_string_for_network_pkl(network_pkl):
    p = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')
    longname = '-'.join(p[max(len(p) - 2, 0):])
    longname = longname.replace('network-snapshot-', '').replace('config', '')
    return '-'.join(longname.split('-')[2:])

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_interpolation_video(network_pkl = None, grid_size=[1,1], png_sequence=False, image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, truncation_psi=1, randomize_noise=False, filename=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    
    if network_pkl == None:
        print('ERROR: Please enter pkl path.')
        sys.exit(1)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)
    if filename is None:
        filename = get_id_string_for_network_pkl(network_pkl) + '-seed-' + str(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = load_pkl(network_pkl)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    print(shape)
    print(len(shape))
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))
    print(all_latents[0].shape)

    
    print("Rendering...\nminibatch_size =", minibatch_size, ", out_shrink =", image_shrink, ", truncation_psi =", truncation_psi, ", randomize_noise =", randomize_noise)
    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8, truncation_psi=truncation_psi, randomize_noise=randomize_noise)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    if png_sequence: 
        result_subdir = "results/videos/" + filename
        Path(result_subdir).mkdir(parents=True, exist_ok=True)
        for png_idx in range(num_frames):
            print('Generating png %d / %d...' % (png_idx, num_frames))
            latents = latents = all_latents[png_idx]
            labels = np.zeros([latents.shape[0], 0], np.float32)
            images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8, truncation_psi=truncation_psi, randomize_noise=randomize_noise)
            misc.save_image_grid(images, os.path.join(result_subdir, '%06d.png' % (png_idx)), [0,255], grid_size)
    else:
        # Generate video.
        import moviepy.editor # pip install moviepy
        result_subdir = "results/videos"
        Path(result_subdir).mkdir(parents=True, exist_ok=True)
        moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, filename + ".mp4"), fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

if __name__ == "__main__":
    main()