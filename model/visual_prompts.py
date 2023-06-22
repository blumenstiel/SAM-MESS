
import cv2
import numpy as np
import random

from scipy import ndimage


def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
    """
    This function generates the point candidates.
    The points are selected using the approach from the paper "Reviving Iterative Training with Mask Guidance for
    Interactive Segmentation".
    Code from:
    https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/master/isegm/data/points_sampler.py
    """
    if full_prob > 0 and random.random() < full_prob:
        return obj_mask

    padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        prob_map = dt.flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        click_indx = np.random.choice(len(prob_map), p=prob_map)
        click_coords = np.unravel_index(click_indx, dt.shape)
        return np.array([click_coords])


def generate_clicks(batch, ignore_labels=None, min_pixel_count=10, max_inputs=100):
    """
    This function adds clicks to the batch. The clicks are added to the batch as a list of points "input_points"
    together with a list of segment labels "input_labels".
    """

    if ignore_labels is None:
        ignore_labels = [255]
    for i, batched_input in enumerate(batch):
        mask = batched_input['sem_seg']
        input_points = []
        input_classes = []
        segment_size = []

        # Add one point for every connected segment
        for c in np.unique(mask):
            if c in ignore_labels:
                # skip ignore labels
                continue

            # get connected segments for class c
            class_segments, num_labels = ndimage.label(mask == c)

            # iterate over segments
            for l in range(1, num_labels + 1):
                segment_mask = class_segments == l

                # check if segment above min_pixel_count
                if np.sum(segment_mask) >= min_pixel_count:
                    # get point candidates
                    candidates = get_point_candidates(segment_mask)

                    # select random point from candidates
                    idx = np.random.randint(0, len(candidates))
                    # add click to input_points (<x,y> format)
                    input_points.append([(candidates[idx][1], candidates[idx][0])])
                    input_classes.append([c])
                    segment_size.append(np.sum(segment_mask))

        # sort clicks by segment size and drop clicks above max_inputs
        sorted_index = np.argsort(segment_size)[::-1]
        input_points = [input_points[i] for i in sorted_index[:max_inputs]]
        input_classes = [input_classes[i] for i in sorted_index[:max_inputs]]

        if len(input_points) == 0:
            # avoid empty input
            input_points = [[(0, 0)]]
            input_classes = [[0]]

        batch[i]['input_points'] = input_points
        batch[i]['input_classes'] = input_classes

    return batch


def generate_boxes(batch, ignore_labels=None, min_pixel_count=10, max_inputs=100):
    """
    This function adds bounding boxes "input_boxes" to the batch together with a list of the labels "input_classes".
    """
    if ignore_labels is None:
        ignore_labels = [255]
    for i, batched_input in enumerate(batch):
        mask = batched_input['sem_seg']
        input_boxes = []
        input_classes = []
        segment_size = []

        # Add one point for every connected segment
        for c in np.unique(mask):
            if c in ignore_labels:
                # skip ignore labels
                continue

            # get connected segments for class c
            class_segments, num_labels = ndimage.label(mask == c)

            # iterate over segments
            for l in range(1, num_labels + 1):
                segment_mask = class_segments == l

                # check if segment above min_pixel_count
                if np.sum(segment_mask) >= min_pixel_count:
                    # get bounding box
                    x, y, w, h = cv2.boundingRect(segment_mask.astype(np.uint8))
                    # add box to input_boxes
                    input_boxes.append([(x, y, x+w, y+h)])
                    input_classes.append([c])
                    segment_size.append(np.sum(segment_mask))

        # sort clicks by segment size and drop clicks above max_inputs
        sorted_index = np.argsort(segment_size)[::-1]
        input_boxes = [input_boxes[i] for i in sorted_index[:max_inputs]]
        input_classes = [input_classes[i] for i in sorted_index[:max_inputs]]

        if len(input_boxes) == 0:
            # avoid empty input
            input_boxes = [[(0, 0, 1, 1)]]
            input_classes = [[0]]

        batch[i]['input_boxes'] = input_boxes
        batch[i]['input_classes'] = input_classes
    return batch
