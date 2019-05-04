from joblib import Parallel, delayed

import argparse, os
from subprocess import call
os.environ['JOBLIB_TEMP_FOLDER'] = "~/tmp"

def load_args():
    ap = argparse.ArgumentParser(description='Paralelize Saliency maps process.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/saliency_frames')
    args = ap.parse_args()
    print(args)
    return args

def extractSaliencyMaps(args, video):

    command = "python main.py test {}/ {}/".format(os.path.join(args.dataset_dir, "frames", video), os.path.join(args.output_path, video))
    print('\n', command)
    call(command, shell=True)

def video_process_finished(args, videos_len, video):
    nb_frames_processed = len(os.listdir(os.path.join(args.output_path, video)))
    if nb_frames_processed != videos_len[video]:
        return False
    else:
        return True

def get_lens(args, videos):
    video_len = {}
    for video in videos:
        video_len[video] = len(os.listdir(os.path.join(args.dataset_dir, "frames", video)))
    return video_len

def main():
    args = load_args()

    videos_to_process = os.listdir(os.path.join(args.dataset_dir, "frames"))
    videos_processed = os.listdir(os.path.join(args.output_path))

    videos_len = get_lens(args, videos_to_process)

    for video in videos_processed:
        if (video_process_finished(args, videos_len, video)):
            videos_to_process.remove(video)
            del videos_len[video]

    ordered_videos_len = sorted(videos_len.items(), key=lambda kv: kv[1])
    sorted_videos_dict = OrderedDict(ordered_videos_len)

    for i, (video, len_value) in enumerate(sorted_videos_dict.items()):
        extractSaliencyMaps(args, video)

#    Parallel(n_jobs=3)(delayed(extractSaliencyMaps)(args, video) for video in videos)

if __name__ == '__main__':
    main()