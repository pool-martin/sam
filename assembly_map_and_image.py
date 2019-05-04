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
    args = ap.parse_args()
    print(args)
    return args

def assemblyImages(args, video):

    print('processing video {}'.format(video))
    output_dir = os.path.join(args.output_path, video)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_frames = os.listdir(os.path.join(args.original_images, video))
    for frame_name in original_frames:
        image_frame = cv2.imread(os.path.join(args.original_images, video, frame_name))
        salience_frame = cv2.imread(os.path.join(args.saliency_maps, video, frame_name), 0)

#        assert(image_frame.shape == salience_frame.shape)

        #The obvious approach
#        salience_frame = cv2.cvtColor(salience_frame, cv2.COLOR_BGR2GRAY)

        #putting salience on blue layer
        image_frame[:,:,0] = salience_frame

        cv2.imwrite(os.path.join(args.output_path, video, frame_name), image_frame.astype(int))

def video_process_finished(args, videos_len, video):
    nb_frames_processed = len(os.listdir(os.path.join(args.output_path, video)))
    if nb_frames_processed != videos_len[video]:
        return False
    else:
        return True

def get_lens(args, videos):
    video_len = {}
    for video in videos:
        video_len[video] = len(os.listdir(os.path.join(args.original_images, video)))
    return video_len

def main():
    args = load_args()

    videos_to_process = os.listdir(args.original_images)
    videos_processed = os.listdir(os.path.join(args.output_path))

    videos_len = get_lens(args, videos_to_process)

    for video in videos_processed:
        if (video_process_finished(args, videos_len, video)):
            videos_to_process.remove(video)
            del videos_len[video]

    ordered_videos_len = sorted(videos_len.items(), key=lambda kv: kv[1])
    sorted_videos_dict = OrderedDict(ordered_videos_len)

    for i, (video, len_value) in enumerate(sorted_videos_dict.items()):
        assemblyImages(args, video)

#    Parallel(n_jobs=3)(delayed(extractSaliencyMaps)(args, video) for video in videos)

if __name__ == '__main__':
    main()