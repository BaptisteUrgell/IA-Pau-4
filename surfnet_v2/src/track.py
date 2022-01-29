from black import out
import numpy as np
import os
from .detection.detect import detect
from .tracking.utils import get_detections_for_video, write_tracking_results_to_file
from .tools.video_readers import IterableFrameReader
from .tools.misc import load_model
from .tracking.trackers import get_tracker
import torch

from .tracking.track_video import track_video, Display

from deep_learning_power_measure.power_measure import experiment, parsers
import time


def main(args, display, demo=False, demo_container=None, video_raw=None, video_name=None):
    if not demo:
        
        if args.device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        device = torch.device(device)

        engine = get_tracker('EKF')

        print('---Loading model...')
        model = load_model(arch=args.arch, model_weights=args.model_weights, device=device)
        print('Model loaded.')

        detector = lambda frame: detect(frame, threshold=args.detection_threshold, model=model)

        transition_variance = np.load(os.path.join(args.noise_covariances_path, 'transition_variance.npy'))
        observation_variance = np.load(os.path.join(args.noise_covariances_path, 'observation_variance.npy'))

        video_filenames = [video_filename for video_filename in os.listdir(args.data_dir) if video_filename.endswith('.mp4')]

        driver = parsers.JsonParser("output_folder")
        exp = experiment.Experiment(driver)
        start = time.time()
        p, q = exp.measure_yourself(period=2)
        
        for video_filename in video_filenames:
            print(f'---Processing {video_filename}')


            print('Tracking...')
            (results,x) = track_video(os.path.join(args.data_dir, video_filename))
            output_filename = os.path.join(args.output_dir, video_filename.split('.')[0] +'.txt')

            output = write_tracking_results_to_file(results,x, ratio_x=1, ratio_y=1, output_filename=output_filename)
            print(output)
    
        q.put(experiment.STOP_MESSAGE)
        driver = parsers.JsonParser("output_folder")
        exp_result = experiment.ExpResults(driver)
        e_time = time.time() - start

        exp_result.print()
        print('Consumption report : \n')
        print('Duration : {} seconds'.format(e_time))
        print('Mean CPU power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("total_cpu_power"))) / (e_time/ 3600),2)))
        print('Mean system power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("psys_power"))) / (e_time/ 3600),2)))
        print('Mean GPU  power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("nvidia_draw_absolute"))) / (e_time / 3600), 2)))
        print('Energy CPU consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("total_cpu_power"))),2)))
        print('Energy system consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("psys_power"))),2)))
        print('Energy GPU  consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("nvidia_draw_absolute"))),2)))

    else:
        driver = parsers.JsonParser("output_folder")
        exp = experiment.Experiment(driver)
        start = time.time()
        p, q = exp.measure_yourself(period=2)
        
        device = torch.device('cpu')
        # engine = get_tracker('EKF')
        # print('---Loading model...')
        # model = load_model(arch='mobilenet_v3_small', model_weights=None, device=device)
        # print('Model loaded.')
        print(f'---Processing Video')
        print('Tracking...')
        (results,x) = track_video('', demo=demo, demo_container=demo_container, video_raw=video_raw, video_name=video_name)
        output_filename = os.path.join('app', 'tmp', 'output', video_name.split('.')[0] +'.txt')

        output = write_tracking_results_to_file(results,x, ratio_x=1, ratio_y=1, output_filename=output_filename)
        
        #print(output)
        
        q.put(experiment.STOP_MESSAGE)
        driver = parsers.JsonParser("output_folder")
        exp_result = experiment.ExpResults(driver)
        e_time = time.time() - start

        exp_result.print()
        print('Consumption report : \n')
        print('Duration : {} seconds'.format(e_time))
        print('Mean CPU power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("total_cpu_power"))) / (e_time/ 3600),2)))
        print('Mean system power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("psys_power"))) / (e_time/ 3600),2)))
        print('Mean GPU  power: {} W'.format(round((experiment.joules_to_wh(exp_result.get_info("nvidia_draw_absolute"))) / (e_time / 3600), 2)))
        print('Energy CPU consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("total_cpu_power"))),2)))
        print('Energy system consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("psys_power"))),2)))
        print('Energy GPU  consumed: {} Wh'.format(round((experiment.joules_to_wh(exp_result.get_info("nvidia_draw_absolute"))),2)))

        
        
        
        
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--detection_threshold', type=float, default=0.3)
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--noise_covariances_path',type=str)
    parser.add_argument('--skip_frames',type=int,default=0)
    parser.add_argument('--output_shape',type=str,default='960,544')
    parser.add_argument('--arch', type=str, default='dla_34')
    parser.add_argument('--display', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--detection_batch_size',type=int,default=1)
    parser.add_argument('--preload_frames', action='store_true', default=False)
    args = parser.parse_args()

    display = None
    if args.display == 0:
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    args.output_shape = tuple(int(s) for s in args.output_shape.split(','))

    main(args, display)
