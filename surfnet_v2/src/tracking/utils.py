from scipy.stats import multivariate_normal
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
import torch
from ..tools.video_readers import TorchIterableFromReader
from time import time
from ..detection.transforms import TransformFrames
from collections import defaultdict

def postprocess_for_api(results, labels, value):
    """ Converts tracking results into json object for API
    """
    result_list = []
    id_list = {}

    for res in results:
        frame_number = res[0]
        box = [res[2], res[3], res[4], res[5]]
        id = res[1]
        label = res[6]
        prob = res[7]
        dict_class = {1: 'Insulating material',
        4: 'Drum',
        2: 'Bottle-shaped',
        3: 'Can-shaped',
        5: 'Other packaging',
        6: 'Tire',
        7: 'Fishing net / cord',
        8: 'Easily namable',
        9: 'Unclear',
        0: 'Sheet / tarp / plastic bag / fragment'}

        # if the id is not already is the results, add a new jsonline
        if id not in id_list:
            id_list[id] = len(result_list)
            result_list.append({"label":dict_class[label],
                                "id": id,
                                "frame_to_box": {str(frame_number): box}})
        else:
            result_list[id_list[id]]["frame_to_box"][str(frame_number)] = box
    return {"detected_trash": result_list}

class GaussianMixture(object):
    def __init__(self, means, covariance, weights):
        self.components = [multivariate_normal(
            mean=mean, cov=covariance) for mean in means]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.pdf(x)
        return result

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.cdf(x)
        return result

def init_trackers(engine, detections, frame_nb, state_variance, observation_variance, delta):
    trackers = []

    for detection in detections:
        tracker_for_detection = engine(frame_nb, detection, state_variance, observation_variance, delta)
        trackers.append(tracker_for_detection)

    return trackers

def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()

def in_frame(position, shape, border=0.02):


    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return x > border*shape_x and x < (1-border)*shape_x and y > border*shape_y and y < (1-border)*shape_y

def gather_filenames_for_video_in_annotations(video, images, data_dir):
    images_for_video = [image for image in images
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    return [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]

def get_detections_for_video(reader, detector, batch_size=16, device=None):

    detections = []
    dataset = TorchIterableFromReader(reader, TransformFrames())
    loader = DataLoader(dataset, batch_size=batch_size)
    average_times = []
    with torch.no_grad():
        for preprocessed_frames in loader:
            time0 = time()
            detections_for_frames = detector(preprocessed_frames.to(device))
            average_times.append(time() - time0)
            for detections_for_frame in detections_for_frames:
                if len(detections_for_frame): detections.append(detections_for_frame)
                else: detections.append(np.array([]))
    print(f'Frame-wise inference time: {batch_size/np.mean(average_times)} fps')
    return detections


def resize_external_detections(detections, ratio):

    for detection_nb in range(len(detections)):
        detection = detections[detection_nb]
        if len(detection):
            detection = np.array(detection)[:,:-1]
            detection[:,0] = (detection[:,0] + detection[:,2])/2
            detection[:,1] = (detection[:,1] + detection[:,3])/2
            detections[detection_nb] = detection[:,:2]/ratio
    return detections


def write_tracking_results_to_file(results,x,ratio_x, ratio_y, output_filename):
    """ writes the output result of a tracking the following format:
    - frame
    - id
    - x_tl, y_tl, w=0, h=0
    - 4x unused=-1
    """
    print("writing to file : "+str(output_filename))
    with open(output_filename, 'w') as output_file:
        for i in range(len(x)):
            output_file.write('{},{} \n'.format(i,x[i]))
        for result in results:
            print(result)
            output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
                                                                result[1]+1,
                                                                ratio_x * result[2],
                                                                ratio_y * result[3],
                                                                ratio_x * result[4],
                                                                ratio_y * result[5],
                                                                result[6],result[7],-1,-1))
    return postprocess_for_api(results, i, x[i])

def read_tracking_results(input_file):
    """ read the input filename and interpret it as tracklets
    i.e. lists of lists
    """
    class_raw_results = np.loadtxt(input_file, delimiter=',',  max_rows=10)
    tracking_raw_results = np.loadtxt(input_file, delimiter=',',skiprows=10)

    if class_raw_results.ndim == 1: class_raw_results = np.expand_dims(class_raw_results,axis=0)
    if tracking_raw_results.ndim == 1: tracking_raw_results = np.expand_dims(tracking_raw_results,axis=0)

    tracklets = defaultdict(list)
    class_labels = []
    class_values = []
    
    for result in class_raw_results:
        class_labels = result[0]
        class_values = result[1]

    for result in tracking_raw_results:
        frame_id = int(result[0])
        track_id = int(result[1])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())
    return (tracklets, class_labels, class_values)

def gather_tracklets(tracklist):
    """ Converts a list of flat tracklets into a list of lists
    """
    tracklets = defaultdict(list)
    for track in tracklist:
        frame_id = track[0]
        track_id = track[1]
        center_x = track[2]
        center_y = track[3]
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())
    return tracklets

class FramesWithInfo:
    def __init__(self, frames, output_shape=None):
        self.frames = frames
        if output_shape is None:
            self.output_shape = frames[0].shape[:-1][::-1]
        else: self.output_shape = output_shape
        self.end = len(frames)
        self.read_head = 0

    def __next__(self):
        if self.read_head < self.end:
            frame = self.frames[self.read_head]
            self.read_head+=1
            return frame

        else:
            raise StopIteration

    def __iter__(self):
        return self
