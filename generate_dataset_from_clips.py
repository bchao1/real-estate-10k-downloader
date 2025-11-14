import cv2
import argparse
import json
import random
import os
import os.path as osp
import imageio
import numpy as np
from decord import VideoReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_clips_folder', required=True)
    parser.add_argument('--clips_folder', required=True)
    parser.add_argument('--clip_txt_folder', required=True)
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--video_width', type=int, default=832)
    parser.add_argument('--video_height', type=int, default=480)
    parser.add_argument('--save_images', action='store_true')

    return parser.parse_args()


def read_clip_txt(clip_txt_folder, clip_id):
    camera_extrinsics = []
    camera_intrinsics = []

    with open(os.path.join(clip_txt_folder, f"{clip_id}.txt"), "r") as f:
        for line in f:
            tokens = line.strip().split(' ')
            if len(tokens) > 1:  # extrinsics lines
                tokens = [float(x) for x in tokens[1:]]  # ignore first timestep token
                # 18 tokens in total. 4 for intrinsics,
                intrinsics = tokens[:4]
                extrinsics = tokens[-12:]
                camera_extrinsics.append(extrinsics)
                camera_intrinsics.append(intrinsics)

    return camera_extrinsics, camera_intrinsics


if __name__ == '__main__':
    args = get_args()

    # make dirs for selected clips
    os.makedirs(args.selected_clips_folder, exist_ok=True)
    os.makedirs(osp.join(args.selected_clips_folder, 'selected_poses'), exist_ok=True)
    os.makedirs(osp.join(args.selected_clips_folder, 'selected_clips'), exist_ok=True)
    if args.save_images:
        os.makedirs(osp.join(args.selected_clips_folder, 'selected_images'), exist_ok=True)

    # processing
    clip_file_names = os.listdir(args.clips_folder)
    print(f"Processing {len(clip_file_names)} clips")
    for clip_file_name in tqdm(clip_file_names):
        clip_id = clip_file_name.split(".")[0]
        video_reader = VideoReader(os.path.join(args.clips_folder, clip_file_name))
        camera_extrinsics, camera_intrinsics = read_clip_txt(args.clip_txt_folder, clip_id)

        assert len(camera_extrinsics) == len(camera_intrinsics) == len(video_reader), \
            "Number of camera extrinsics and intrinsics must match number of frames in video"

        # sample frame indices
        total_frames = len(video_reader)
        cropped_length = args.num_frames * args.sample_stride
        start_frame_idx = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_idx = min(start_frame_idx + cropped_length, total_frames)
        assert end_frame_idx - start_frame_idx >= args.num_frames

        frame_indices = np.linspace(start_frame_idx, end_frame_idx - 1, args.num_frames, dtype=int)

        # write selected poses
        with open(osp.join(args.selected_clips_folder, 'selected_poses', f'{clip_id}.txt'), 'w') as f:
            for frame_idx in frame_indices:
                poses_info = camera_intrinsics[frame_idx] + camera_extrinsics[frame_idx]
                f.write(' '.join([str(x) for x in poses_info]) + '\n')

        # write selected clips and images
        video_batch = video_reader.get_batch(frame_indices).asnumpy()
        video_batch = [cv2.resize(x, dsize=(args.video_width, args.video_height)) for x in video_batch]

        imageio.mimsave(
            osp.join(args.selected_clips_folder, 'selected_clips', f'{clip_id}.mp4'),
            video_batch,
            fps=8
        )

        if args.save_images:
            os.makedirs(osp.join(args.selected_clips_folder, 'selected_images', clip_id), exist_ok=True)
            for image_idx, image in zip(frame_indices, video_batch):
                image_save_path = osp.join(
                    args.selected_clips_folder,
                    'selected_images',
                    clip_id,
                    f'{str(image_idx).zfill(6)}.jpg'
                )
                cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
