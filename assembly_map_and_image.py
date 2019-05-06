from joblib import Parallel, delayed

import argparse, os
from subprocess import call
from collections import OrderedDict
import cv2
os.environ['JOBLIB_TEMP_FOLDER'] = "~/tmp"

def load_args():
    ap = argparse.ArgumentParser(description='Paralelize Saliency maps process.')
    ap.add_argument('-sm', '--saliency-maps',
                                    dest='saliency_maps',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/saliency_frames')
    ap.add_argument('-oi', '--original-images',
                                    dest='original_images',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/frames')
    ap.add_argument('-op', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/assembly_frames')
    ap.add_argument('-pp', '--parallel-process',
                                    dest='parallel_process',
                                    help='qtd of parallel videos processed at same time.',
                                    type=int, default=10)
    ap.add_argument('-s', '--split',
                                    dest='split',
                                    help='split to process',
                                    type=str, default='s1_a')
    args = ap.parse_args()
    print(args)
    return args

def assemblyImages(args, video):

    print('processing video {}'.format(video))
    output_dir = os.path.join(args.output_path, video)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

#    original_frames = os.listdir(os.path.join(args.original_images, video))
    saliency_frames = os.listdir(os.path.join(args.saliency_maps, video))
    for frame_name in saliency_frames:
        image_frame = cv2.imread(os.path.join(args.original_images, video, frame_name))
        salience_frame = cv2.imread(os.path.join(args.saliency_maps, video, frame_name), 0)

#        assert(image_frame.shape == salience_frame.shape)

        #The obvious approach
#        salience_frame = cv2.cvtColor(salience_frame, cv2.COLOR_BGR2GRAY)

        #putting salience on blue layer
        image_frame[:,:,0] = salience_frame

        cv2.imwrite(os.path.join(args.output_path, video, frame_name), image_frame.astype(int))

def get_dataset(split):
    folds_path = "/Exp/2kporn/splits/{}/2D/1_fps/opencv/".format(split)
    file_names = []
    sets = ['network_training_set.txt', 'network_validation_set.txt', 'test_set.txt']
    for sset in sets:
        set_file = os.path.join(folds_path, sset)
        with open(set_file) as fi:
            file_names.extend(fi.read().splitlines())
    return file_names

def video_process_finished(args, videos_len, dataset_bag, video):
    video_frames_to_process = get_video_frames(args.split, dataset_bag, video)
    video_frames_processed = os.listdir(os.path.join(args.output_path, video))

    missed_frames_to_process = [f for f in video_frames_to_process if f+'.jpg' not in video_frames_processed]
    if len(missed_frames_to_process) > 0:
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

    videos_to_process = os.listdir(args.original_images)
    videos_processed = os.listdir(os.path.join(args.output_path))

    dataset_bag = get_dataset(args.split)
    videos_len = get_lens(args, dataset_bag, videos_to_process)

    for video in videos_to_process:
        if (video_process_finished(args, videos_len, dataset_bag, video)):
            del videos_len[video]

    ordered_videos_len = sorted(videos_len.items(), key=lambda kv: kv[1])
    sorted_videos_dict = OrderedDict(ordered_videos_len)

    if(args.parallel_process == 1):
        for i, (video, len_value) in enumerate(sorted_videos_dict.items()):
                assemblyImages(args, video)
    else:
        Parallel(n_jobs=args.parallel_process)(delayed(assemblyImages)(args, video) for video, len_video in sorted_videos_dict.items())

if __name__ == '__main__':
    main()