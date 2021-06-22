# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import os.path as osp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.detection import Detection
from utils.tracker import Tracker
from utils.config import cfg
from GMMOT.graph_encoder import ReidEncoder

def gather_sequence_info(sequence_dir, npy_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = npy_file
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "img_dir": image_dir
    }
    return seq_info
def encode_newfeat(npy_file, checkpoint_dir):
    # encode new appearance feature from original reid feature
    npydata = np.load(npy_file)
    reid = torch.Tensor(npydata[:,10:])
    print(reid.shape)
    class Net0(nn.Module):
        def __init__(self):
            super(Net0, self).__init__()
            self.reid_enc = ReidEncoder()
            self.cross_graph = nn.Linear(512, 512)
        def forward(self, x):
            x = self.reid_enc(x)
            return x
    with torch.no_grad():
        model = Net0()
        model.eval()
        params_path = os.path.join(checkpoint_dir, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path),strict=False)
        feat_new = model(reid)
        print(feat_new.shape)
        npydata[:,10:] = feat_new
        return npydata

def create_detections(detection_mat, frame_idx, w_img=0,h_img=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        '''
        bbx2 = bbox[0]+bbox[2] if bbox[0]+bbox[2]<=w_img else w_img
        bby2 = bbox[1]+bbox[3] if bbox[1]+bbox[3]<=h_img else h_img
        bbx1 = bbox[0] if bbox[0]>=0 else 0.0
        bby1 = bbox[1] if bbox[1]>=0 else 0.0
        bbox[0] = bbx1
        bbox[1] = bby1
        bbox[2] = bbx2 - bbx1
        bbox[3] = bby2 - bby1
        '''
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def ECC(src0, dst0, warp_mode = cv2.MOTION_EUCLIDEAN, eps = 1e-5,
        max_iter = 100):
    src = cv2.cvtColor(src0, cv2.COLOR_BGR2GRAY)
    dst = cv2.cvtColor(dst0, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    (cc, warp_matrix) = cv2.findTransformECC (src, dst, warp_matrix, warp_mode, criteria, None, 1)
    return warp_matrix

class WarpMatrix():
    def __init__(self, seq_info):
        self.mat = []
        for i in range(1, seq_info["max_frame_idx"]):
            print('Generating warp_mat '+seq_info["sequence_name"]+ " frame %05d" %i)
            image_1 = cv2.imread(os.path.join(seq_info["img_dir"], "%06d"%(i) +".jpg"))
            image_2 = cv2.imread(os.path.join(seq_info["img_dir"], "%06d"%(i+1) +".jpg"))
            warp_mat = ECC(image_1,image_2)
            self.mat.append(warp_mat)

def run(sequence_dir, detection_file, output_file, max_age, n_init, reid_thr, checkpoint_dir):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    new_npy = encode_newfeat(detection_file, checkpoint_dir)
    seq_info = gather_sequence_info(sequence_dir, new_npy)
    tracker = Tracker(max_age=max_age, n_init=n_init,reid_thr=reid_thr)
    results = []
    if not osp.exists(os.path.join("warp_mat", "%s.npy" %seq_info["sequence_name"])):
        if not osp.exists("./warp_mat"):
            os.system('mkdir ./warp_mat')
        warp_matrix = np.array(WarpMatrix(seq_info).mat)
        output_filename = os.path.join("warp_mat", "%s.npy" %seq_info["sequence_name"])
        np.save(
            output_filename, warp_matrix, allow_pickle=False)
    else:
        warp_matrix = np.load(os.path.join("warp_mat", "%s.npy" %seq_info["sequence_name"]))
    def frame_callback(frame_idx):
        print("Processing %s"%seq_info["sequence_name"], "frame %05d" %frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, w_img=seq_info["image_size"][1],h_img=seq_info["image_size"][0])
        
        # Update tracker.
        tracker.predict(warp_matrix[frame_idx-2])
        tracker.update(detections, seq_info["sequence_name"], frame_idx, checkpoint_dir)

        # Store results.
        for track in tracker.tracks:
            if track.time_since_update >= 1:
                continue
            bbox = track.to_tlwh2()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    frame_idx = seq_info["min_frame_idx"]
    while frame_idx <= seq_info["max_frame_idx"]:
        frame_callback(frame_idx)
        frame_idx += 1

    # Store results.
    output_path = os.path.dirname(output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Learnable Graph Matching for MOT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--checkpoint_dir", help="Path to checkpoint dir.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--max_age", help="The maximum frames to delete a tracklet.",
        default=100, type=int)
    parser.add_argument(
        "--n_init", help="n_init",
        default=1, type=int)
    parser.add_argument(
        "--reid_thr", help="Cosine similarity threshold of ReID features.",
        default=0.6, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.max_age, args.n_init, args.reid_thr, args.checkpoint_dir)
