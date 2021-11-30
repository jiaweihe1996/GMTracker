import numpy as np 
import csv
from utils.config import cfg
import argparse
import os
def bbox_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    # bbox1 = [float(x) for x in bbox1]
    # bbox2 = [float(x) for x in bbox2]

    # (x0_1, y0_1, x1_1, y1_1) = bbox1
    # (x0_2, y0_2, x1_2, y1_2) = bbox2
    x0_1 = float(bbox1[0])
    y0_1 = float(bbox1[1])
    x1_1 = float(bbox1[0])+float(bbox1[2])
    y1_1 = float(bbox1[1])+float(bbox1[3])
    x0_2 = float(bbox2[0])
    y0_2 = float(bbox2[1])
    x1_2 = float(bbox2[0])+float(bbox2[2])
    y1_2 = float(bbox2[1])+float(bbox2[3])
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
def run(input,output):
    if not os.path.exists(output):
        os.system('mkdir -p %s'%output)
    for seq_det in os.listdir(input):

        print('post-processing {:s}'.format(seq_det))
        file_name = os.path.join(input,seq_det)

        results = []
        with open(file_name, 'r') as fi:
            reader = csv.reader(fi)
            mottxt = [row for row in reader]
            print('original: %d objects'%len(mottxt))
            track_max = np.array([int(row[1]) for row in mottxt]).max()
            for i in range(1, track_max+1):
                tracklet = []
                for row in mottxt:
                    # print(row[0])
                    if int(row[1]) == i:
                        tracklet.append(row)
                if len(tracklet)>1:
                    frame_in_tracklet_i = [int(row[0]) for row in tracklet]
                    min_frame_tracklet_i = np.array(frame_in_tracklet_i).min()
                    max_frame_tracklet_i = np.array(frame_in_tracklet_i).max()
                    for j in range(min_frame_tracklet_i, max_frame_tracklet_i+1):
                        if j in frame_in_tracklet_i:
                            for row in tracklet:
                                if int(row[0])==j:
                                    # row_i = row 
                                    results.append(row)
                        if j not in frame_in_tracklet_i:
                            for row in tracklet:
                                if int(row[0])<j:
                                    frame_last = row
                                if int(row[0])>j:
                                    frame_later = row
                                    interval = int(frame_later[0]) - int(frame_last[0])
                                    interval_j = j - int(frame_last[0])
                                    bbox_l = float(frame_last[2]) + interval_j/interval*(float(frame_later[2])-float(frame_last[2]))
                                    bbox_t = float(frame_last[3]) + interval_j/interval*(float(frame_later[3])-float(frame_last[3]))
                                    bbox_last_r = float(frame_last[2]) + float(frame_last[4])
                                    bbox_later_r = float(frame_later[2]) + float(frame_later[4])
                                    bbox_last_b = float(frame_last[3]) + float(frame_last[5])
                                    bbox_later_b = float(frame_later[3]) + float(frame_later[5])                                
                                    bbox_r = bbox_last_r + interval_j/interval*(bbox_later_r-bbox_last_r)
                                    bbox_b = bbox_last_b + interval_j/interval*(bbox_later_b-bbox_last_b)
                                    results.append([str(j),str(i),str(bbox_l),str(bbox_t),str(bbox_r-bbox_l),str(bbox_b-bbox_t)])
                                    break
        print('after post-process: %s objects'%len(results))
        f = open(os.path.join(output,seq_det), 'w')
        for row in results:
            print('%s,%s,%s,%s,%s,%s,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)



def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Post-process with tracklet linear interpolation")
    parser.add_argument(
        "--input_dir", help="Path to online result dir",
        default=cfg.RESULTS_VAL_DIR)
    parser.add_argument(
        "--output_dir", help="Path to output result dir.", default=None,
        required=True)  
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.input_dir, args.output_dir)
    run(args.input_dir, args.output_dir)
