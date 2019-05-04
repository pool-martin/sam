from joblib import Parallel, delayed

import argparse, os
from subprocess import call

def load_args():
    ap = argparse.ArgumentParser(description='Paralelize Saliency maps process.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/frames/')
    args = ap.parse_args()
    print(args)
    return args

def extractSaliencyMaps(args, video):

    command = "python main.py test {}".format(os.path.join(args.dataset_dir, "frames", video))
    print('\n', command)
    call(command, shell=True)

def extract(args, all_set):

    class_types = ['Porn', 'NonPorn']
    video_frames = {}

    for class_type in class_types:
        for i in range(1, 1001):
            video = 'v{}{}'.format(class_type, str(i).zfill(6))

    Parallel(n_jobs=10)(delayed(extractSaliencyMaps)(args, video) for video in video_frames.keys())

    

def main():
    args = load_args()

    videos = os.listdir(os.path.join(args.dataset_dir, "frames")) 
    extract(args, videos)

if __name__ == '__main__':
    main()