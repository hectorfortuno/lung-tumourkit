import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from skimage.morphology import reconstruction

import sys
sys.path.append(r"../../")
from misc.utils import get_bounding_box, remove_small_objects

import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop


####
def __proc_np_hv(pred, h_value=0.5, k_value=0.4):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """

    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    # blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = np.array(blb_raw >= h_value, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=50)  # 10
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # overall = np.array(overall >= 0.4, dtype=np.int32)
    overall = np.array(overall >= k_value, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=50)          # 10

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred, dist, marker, blb


def __proc_np_hv_iterative(pred, h_value=0.5, k_value=0.4, h_min_value=0.1):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """

    # Initialize
    markers = np.uint8(np.zeros(pred.shape[:2]))
    blbs = np.uint8(np.zeros(pred.shape[:2]))
    dists = np.zeros(pred.shape[:2])

    # Results for each "h"
    h_values = list(np.arange(h_min_value, h_value, 0.1)) + [h_value]
    for h in h_values[::-1]:

        # Post-process with a value of h
        pred_inst_h, dist_h, marker_h, blb_h = __proc_np_hv(pred, h, k_value)

        # Obtain the new elements of this level and add them to the previous predicted instances
        marker_h = np.uint8(marker_h > 0)
        old_markers = reconstruction(np.minimum(markers, marker_h), marker_h)
        new_markers = marker_h - old_markers
        markers += np.uint8(new_markers)

        old_blbs = reconstruction(np.minimum(blbs, blb_h), blb_h)
        new_blbs = blb_h - old_blbs
        blbs += np.uint8(new_blbs)

        dist_h[new_blbs == 0] = 0
        dists += dist_h

    # Split cells
    markers = measurements.label(markers)[0]
    pred_inst = watershed(dists, markers=markers, mask=blbs)

    return pred_inst


####
def process(pred_map, prob_map, logits_map, pred_map_stroma=None, nr_types=None, return_centroids=False, h=None, k=None):
    """Post processing script for image tiles.

    Args:
        pred_map: combined output of tp, np and hv branches, in the same order
        prob_map: probabilities of the pixels from tp branch (all the classes)
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """

    # Number of classes (including background)
    if nr_types is not None and pred_map_stroma is not None:
        nr_types += 1

    # Predictions of the HoverNet
    if nr_types is not None:
        pred_type = pred_map[..., :1]       # TP branch result
        pred_inst = pred_map[..., 1:]       # NP and HV branches result
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    # Process NP and HV branches to obtain instances of cells
    pred_inst = np.squeeze(pred_inst)
    #pred_inst = __proc_np_hv(pred_inst, h, k)[0]
    pred_inst = __proc_np_hv_iterative(pred_inst, h, k)

    # Information of each predicted cell
    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id

            # Bounding box of the cell
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            # inst_map = inst_map[inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]]
            inst_map = inst_map[rmin:rmax, cmin:cmax]
            inst_map = inst_map.astype(np.uint8)

            # Contour of the cell
            inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points don't make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_contour[:, 0] += cmin  # inst_bbox[0][1]   # X
            inst_contour[:, 1] += rmin  # inst_bbox[0][0]   # Y

            # Centroid of the cell
            inst_moment = cv2.moments(inst_map)
            inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                             (inst_moment["m01"] / inst_moment["m00"]),]
            inst_centroid = np.array(inst_centroid)
            inst_centroid[0] += cmin    # inst_bbox[0][1]   # X
            inst_centroid[1] += rmin    # inst_bbox[0][0]   # Y

            # Initialize cell information
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "prob1": None, # Added probability of positive
                "type": None,
            }
            if nr_types is not None and int(nr_types) > 2:
                for k in range(0, int(nr_types)):
                    inst_info_dict[inst_id]["prob" + str(k)] = None # Initialized probabilities of all classes

    """
    # PROPORTION
    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):

            # Cell prediction
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

            # Stroma mask above current prediction
            if pred_map_stroma is not None:
                inst_stroma_map_crop = pred_map_stroma[rmin:rmax, cmin:cmax]
                stroma_mask = (inst_stroma_map_crop >= 0.5)
                inst_type_crop[stroma_mask] = nr_types-1     # stroma label

            # TP prediction only on the cell, not background
            inst_map_crop = (inst_map_crop == inst_id)  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]

            # Compute type of cell and probability of each class
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

            # Compute probability of positive (binary)
            if not 2 in type_dict and not 1 in type_dict:
                prob1 = 0
            elif not 1 in type_dict:
                prob1 = 1
            elif not 2 in type_dict:
                prob1 = 0
            else:
                prob1 = type_dict[2] / (type_dict[1] + type_dict[2])
            inst_info_dict[inst_id]["prob1"] = float(prob1)

            # Multi-class probabilities
            if int(nr_types) > 2:
                total_sum = np.sum(list(type_dict.values()))
                # if 0 in type_dict:
                #     total_sum -= type_dict[0]
                for k in range(0, int(nr_types)):
                    if not k in type_dict:
                        type_count = 0
                    else:
                        type_count = type_dict[k]
                    inst_info_dict[inst_id]["prob" + str(k)] = float(type_count / (total_sum + 1.0e-6))
    """

    # MEAN
    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):

            # Cell prediction
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_prob_crop = prob_map[rmin:rmax, cmin:cmax]
            inst_logits_crop = logits_map[rmin:rmax, cmin:cmax]

            # Stroma mask above current prediction
            if pred_map_stroma is not None:
                inst_stroma_map_crop = pred_map_stroma[rmin:rmax, cmin:cmax]
                stroma_mask = (inst_stroma_map_crop >= 0.5)
                inst_type_crop[stroma_mask] = nr_types - 1  # stroma label
                inst_stroma_map_crop = np.expand_dims(inst_stroma_map_crop, axis=-1)
                inst_prob_crop = np.concatenate((inst_prob_crop, inst_stroma_map_crop), axis=-1)

            # TP prediction only on the cell, not background
            inst_map_crop = (inst_map_crop == inst_id)  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            inst_prob = inst_prob_crop[inst_map_crop]
            inst_logits = inst_logits_crop[inst_map_crop]

            # Compute type of cell and probability of each class
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}

            # Probabilities and logits (from mean value)
            type_prob_mean = np.mean(inst_prob, axis=0)
            type_prob = type_prob_mean / np.sum(type_prob_mean)
            type_logits_mean = np.mean(inst_logits, axis=0)
            type_logits = type_logits_mean / np.sum(type_logits_mean)

            # Type of the cell
            max_id = np.argmax(type_prob)
            inst_info_dict[inst_id]["type"] = int(max_id)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob[max_id])

            # Compute probability of positive (binary) TODO: add background probability
            if not 2 in type_dict and not 1 in type_dict:
                prob1 = 0
            elif not 1 in type_dict:
                prob1 = 1
            elif not 2 in type_dict:
                prob1 = 0
            else:
                prob1 = type_dict[2] / (type_dict[1] + type_dict[2])
            inst_info_dict[inst_id]["prob1"] = float(prob1)

            # Multi-class probabilities
            if int(nr_types) > 2:
                for k in range(0, int(nr_types)):
                    inst_info_dict[inst_id]["prob" + str(k)] = float(type_prob[k])
                    # inst_info_dict[inst_id]["logit" + str(k)] = float(type_logits[k])

    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict
