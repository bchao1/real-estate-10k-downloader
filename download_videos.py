import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video_folder', required=True)
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Number of concurrent downloads to run (default: 1).',
    )
    args = parser.parse_args()
    with open(os.path.join(args.video_folder, 'video_files.json'), 'r') as f:
        video_files = json.load(f)
    print(f'There are {len(video_files)} videos to download')

    os.makedirs(args.video_folder, exist_ok=True)

    def download_video(video_id: str) -> None:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': os.path.join(args.video_folder, f'{video_id}.%(ext)s'),
            'cookiefile': '../data/yt_cookies.txt'
        }
        url = f'https://www.youtube.com/watch?v={video_id}'
        if os.path.exists(os.path.join(args.video_folder, f'{video_id}.mp4')):
            print(f'Video {video_id} already exists')
            return
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f'Failed to download {video_id}: {e}')
            return

    if args.jobs <= 1:
        for video_id in video_files:
            download_video(video_id)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(download_video, video_id): video_id for video_id in video_files
            }
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'Failed to download {video_id}: {exc}')