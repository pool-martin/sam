from joblib import Parallel, delayed

import argparse, os
from subprocess import call
from collections import OrderedDict
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
    ap.add_argument('-fs', '--filter-size',
                                    dest='filter_size',
                                    help='max video size to process.',
                                    type=int, default=60000)
    ap.add_argument('-pp', '--parallel-process',
                                    dest='parallel_process',
                                    help='qtd of parallel videos processed at same time.',
                                    type=int, default=10)
    ap.add_argument('-so', '--sort-order',
                                    dest='sort_order',
                                    help='order the process by (a) upward or (d) downward.',
                                    type=str, default='a')
    ap.add_argument('-cp', '--class-to-process',
                                    dest='class_to_process',
                                    help='Process only (Porn) or (NonPorn).',
                                    type=str, default='')
    ap.add_argument('-s', '--split',
                                    dest='split',
                                    help='split to process',
                                    type=str, default='s1_a')

    args = ap.parse_args()
    print(args)
    return args

def extractSaliencyMaps(args, video):

    command = "python main.py test {}/ {}/ /Exp/2kporn/splits/2D/{}/2D/1_fps/opencv/ {}".format(os.path.join(args.dataset_dir, "frames", video), os.path.join(args.output_path, video), args.split, video)
    print('\n', command)
    call(command, shell=True)

def get_dataset(split):
    folds_path = "/Exp/2kporn/splits/2D/{}/2D/1_fps/opencv/".format(split)
    file_names = []
    sets = ['network_training_set.txt', 'network_validation_set.txt', 'test_set.txt']
    for sset in sets:
        set_file = os.path.join(folds_path, sset)
        with open(set_file) as fi:
            file_names.extend(fi.read().splitlines())
    return file_names

def video_process_finished(args, videos_len, video):
    nb_frames_processed = len(os.listdir(os.path.join(args.output_path, video)))
    if nb_frames_processed != videos_len[video]:
        return False
    else:
        return True

def get_video_frames(split, dataset_bag, video):

    # vPorn000002_1_0 vPorn000002_1_151.jpg
#    file_names = [os.path.join(f.split('_')[0], '{}.jpg'.format(f)) for f in dataset_bag]
    video_frames = [f for f in dataset_bag if video in f]
    return video_frames

def get_lens(args, dataset_bag, videos):
    video_len = {}
    for video in videos:
#        video_len[video] = len(os.listdir(os.path.join(args.dataset_dir, "frames", video)))
        video_len[video] = len(get_video_frames(args.split, dataset_bag, video))
    return video_len

def main():
    args = load_args()

    videos_to_process = os.listdir(os.path.join(args.dataset_dir, "frames"))
    videos_processed = os.listdir(os.path.join(args.output_path))

    dataset_bag = get_dataset(args.split)
    videos_len = get_lens(args, dataset_bag, videos_to_process)

    for video in videos_processed:
        if (video_process_finished(args, videos_len, video)):
            videos_to_process.remove(video)
            del videos_len[video]

    print("videos to process before filter by len: {} in ({}) order".format(len(videos_len), args.sort_order))
    for video in videos_to_process:
        if(args.sort_order == 'a'):
            if(videos_len[video] > args.filter_size):
                del videos_len[video]
        else:
            if(videos_len[video] < args.filter_size):
                del videos_len[video]

    print("videos to process after filter by len: {} in ({}) order".format(len(videos_len), args.sort_order))

    ordered_videos_len = sorted(videos_len.items(), key=lambda kv: kv[1], reverse= (True if args.sort_order == 'd' else False))
    sorted_videos_dict = OrderedDict(ordered_videos_len)

    if(args.parallel_process == 1):
        for i, (video, len_value) in enumerate(sorted_videos_dict.items()):
            extractSaliencyMaps(args, video)
    else:
        Parallel(n_jobs=args.parallel_process)(delayed(extractSaliencyMaps)(args, video) for video, len_video in sorted_videos_dict.items())

if __name__ == '__main__':
    main()