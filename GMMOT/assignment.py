# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import kalman_filter
import torch
from GMMOT.gmatching import *
from utils.kalman_filter import *
from utils.build_graphs import *
from utils.config import cfg
import os
import torchvision
INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    #print(detection_indices,track_indices)
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
        
    return matches, unmatched_tracks, unmatched_detections



def quadratic_matching(
        tracks, detections, track_indices=None,
        detection_indices=None,reid_thr=0.8,seq_name=None,ckp_dir=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    
    dets = np.array([detections[i].feature for i in detection_indices])
    tras = np.array([tracks[i].mov_ave for i in track_indices])
    tra = torch.Tensor(tras)
    det = torch.Tensor(dets)
    det_geos = np.array([[detections[i].tlwh[0],detections[i].tlwh[1],detections[i].tlwh[2]+detections[i].tlwh[0],detections[i].tlwh[3]+detections[i].tlwh[1]] for i in detection_indices])
    det_geo = torch.Tensor(det_geos)

    tra_means = np.array([[to_tlwh(tracks[i].mean[0:4])[0],to_tlwh(tracks[i].mean[0:4])[1],to_tlwh(tracks[i].mean[0:4])[0]+to_tlwh(tracks[i].mean[0:4])[2],to_tlwh(tracks[i].mean[0:4])[1]+to_tlwh(tracks[i].mean[0:4])[3]] for i in track_indices])
    tra_geo = torch.Tensor(tra_means)
    iou = torchvision.ops.box_iou(tra_geo,det_geo)
    data1 = tra
    data2 = det
    kf_gate = gate(
            kalman_filter.KalmanFilter(), tracks, detections, track_indices,
            detection_indices)
    _, _, start_src, end_src = gh(data1.shape[0])
    _, _, start_tgt, end_tgt = gh(data2.shape[0])
    data1 = data1.t().unsqueeze(0)
    data2 = data2.t().unsqueeze(0)
    start_src = torch.tensor(start_src)
    end_src = torch.tensor(end_src)
    start_tgt = torch.tensor(start_tgt)
    end_tgt = torch.tensor(end_tgt)

    with torch.no_grad():
        graphnet = GraphNet()
        params_path = os.path.join(ckp_dir, f"params.pt")
        graphnet.load_state_dict(torch.load(params_path), strict=False)
        if iou.shape[0] >= iou.shape[1]:
            indices, thr_flag = graphnet.forward(data1, data2, kf_gate, reid_thr, iou, start_src, end_src, start_tgt, end_tgt, seq_name, inverse_flag=False)
        if iou.shape[0] < iou.shape[1]:
            indices, thr_flag = graphnet.forward(data2, data1, kf_gate.T, reid_thr, iou.t(), start_tgt, end_tgt, start_src, end_src, seq_name, inverse_flag=True)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        
        if thr_flag[row, col] == 1:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:  
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def graph_matching(
        max_age, tracks, detections,
        track_indices=None, detection_indices=None,reid_thr=0.8,seq_name=None,ckp_dir=None):

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    matches = []
    #for level in range(cascade_depth):
    if len(unmatched_detections) == 0:  # No detections left
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        print('No detections left')
        return matches, unmatched_tracks, unmatched_detections
    track_indices_l = [
        k for k in track_indices
        if tracks[k].time_since_update < 1 + max_age
    ]
    if len(track_indices_l) == 0:  # Nothing to match at this level
        print('Nothing to match at this level')
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections
    matches_l, _, unmatched_detections = \
        quadratic_matching(
            tracks, detections,
            track_indices_l, unmatched_detections,reid_thr,seq_name,ckp_dir)
    matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def gate(
        kf, tracks, detections, track_indices, detection_indices, only_position=False):
  
    cost = np.zeros((len(track_indices),len(detection_indices)),dtype=int)
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost[row, gating_distance > gating_threshold] = -1
    return cost