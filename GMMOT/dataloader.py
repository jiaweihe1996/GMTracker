import torch
import torchvision
from GMMOT.config import cfg
import numpy as np
import random
from torch_geometric.data import Data
import csv
import os
import cv2
import gc
import re
from utils.kalman_filter import KalmanFilter
class MOTGraph(torch.utils.data.Dataset):
    def __init__(self):
        self.mode = cfg.TRAIN.MODE
        self.kf = KalmanFilter()
        self.framelist = []
        if self.mode == 'all':
            self.videolist = cfg.DATA.MOT15NAME + cfg.DATA.MOT17ALLNAME
            self.num = cfg.DATA.MOT15FRAMENUM + cfg.DATA.MOT17ALLFRAMENUM
            self.framesum = np.sum(self.num) - 2*len(cfg.DATA.MOT15FRAMENUM + cfg.DATA.MOT17ALLFRAMENUM)

            for i in range(len(cfg.DATA.MOT15NAME) + len(cfg.DATA.MOT17ALLNAME)):
                indx = range(3, self.num[i]+1)
                #[self.videolist[i]+'/'+'%06d'%inx for inx in indx]
                self.framelist += [self.videolist[i]+'/'+'%06d'%inx for inx in indx]

        elif self.mode == 'trainval':
            self.videolist = cfg.DATA.MOT15NAME + cfg.DATA.MOT17NAME
            self.num = cfg.DATA.MOT15FRAMENUM + cfg.DATA.MOT17FRAMENUM
            self.framesum = np.sum(self.num) - 2*len(cfg.DATA.MOT15FRAMENUM + cfg.DATA.MOT17FRAMENUM)
            
            for i in range(len(cfg.DATA.MOT15NAME) + len(cfg.DATA.MOT17NAME)):
                indx = range(3, self.num[i]+1)
                #[self.videolist[i]+'/'+'%06d'%inx for inx in indx]
                self.framelist += [self.videolist[i]+'/'+'%06d'%inx for inx in indx]
        elif self.mode == 'onlytrain':
            self.videolist = cfg.DATA.MOT17NAME
            self.num = cfg.DATA.MOT17FRAMENUM
            self.framesum = np.sum(self.num) - 2*len(cfg.DATA.MOT17FRAMENUM)
            
            for i in range(len(cfg.DATA.MOT17NAME)):
                indx = range(3, self.num[i]+1)
                #[self.videolist[i]+'/'+'%06d'%inx for inx in indx]
                self.framelist += [self.videolist[i]+'/'+'%06d'%inx for inx in indx]
        else:
            raise RuntimeError('Mode error!')

        if len(self.framelist) != self.framesum:
            raise RuntimeError('len(self.framelist) != self.framesum')
        ## delete invalid frames
        delete_list = []
        for index, videoname in enumerate(self.videolist):
            gt_file = cfg.DATA.PATH_TO_DATA_DIR + videoname + '/gt/gt.txt'
            if videoname in cfg.DATA.MOT15NAME:
                with open(gt_file, 'r') as fi:
                    reader = csv.reader(fi)
                    frame = set([int(row[0]) if row[6] != '0' else 0 for row in reader]) - {0}
            else:
                if videoname == 'MOT17-04':
                    print(videoname)
                    with open(gt_file, 'r') as fi:
                        reader = csv.reader(fi)
                        frame = set([int(row[0]) if (row[6] != '0' and (row[7] == '1' or row[7] == '2') and float(row[8])>=0.2) and int(row[0])%5==3 else 0 for row in reader]) - {0}
                    print(len(frame))
                else:
                    with open(gt_file, 'r') as fi:
                        reader = csv.reader(fi)
                        frame = set([int(row[0]) if (row[6] != '0' and (row[7] == '1' or row[7] == '2') and float(row[8])>=0.2) else 0 for row in reader]) - {0}

            for i in range(self.num[index]):
                if i not in frame:
                    delete_list.append(videoname+'/'+'%06d'%i)
        self.framelist = list(set(self.framelist)-set(delete_list))


    def __len__(self):
        return len(self.framelist)

    def construct_graph(self, features, geos, feat_no_ma):

        if len(feat_no_ma) == 0:
            fnm = torch.tensor([])
        else:
            fnm = [torch.Tensor(feat_no_ma[i]) for i in range(len(feat_no_ma))]
            # print(fnm.shape[])
        x = torch.Tensor(features)
        geo = torch.Tensor(geos)
        n_node = x.shape[0]
        ################################
        A = torch.ones(n_node,n_node) - torch.eye(n_node,n_node)
        edge_list = [[], []]
        #print(n_node)
        for i in range(n_node):
            for j in range(n_node):
                if A[i, j] == 1:
                    edge_list[0].append(i)
                    edge_list[1].append(j)
        ################################
        edge_index = torch.tensor(edge_list)
        #print(edge_index)

        graph = Data(x=x, geo=geo, edge_index=edge_index,fnm=fnm)
        return graph

    def gather_sequence_info(self, sequence_dir, detection_file):

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}
        detections = None
        if detection_file is not None:
            detections = np.load(detection_file)

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

        feature_dim = detections.shape[1] - 9 if detections is not None else 0
        seq_info = {
            "sequence_name": os.path.basename(sequence_dir),
            "image_filenames": image_filenames,
            "detections": detections,
            "image_size": image_size,
            "min_frame_idx": min_frame_idx,
            "max_frame_idx": max_frame_idx,
            "feature_dim": feature_dim,
            "update_ms": update_ms
        }
        return seq_info
    def moving_average(self, feat, saved_ma, alpha):
        if len(saved_ma) == 0:
            ema = feat
        else:
            ema = saved_ma * alpha + feat * (1 - alpha)
        return ema
    def to_xyah(self, tlwh, w_img, h_img):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.zeros(4)
        bbx2 = tlwh[0]+tlwh[2] if tlwh[0]+tlwh[2]<=w_img else w_img
        bby2 = tlwh[1]+tlwh[3] if tlwh[1]+tlwh[3]<=h_img else h_img
        bbx1 = tlwh[0] if tlwh[0]>=0 else 0.0
        bby1 = tlwh[1] if tlwh[1]>=0 else 0.0
        ret[0] = (bbx1 + bbx2) / 2
        ret[1] = (bby1 + bby2) / 2
        ret[2] = (bbx2 - bbx1) / (bby2 - bby1)
        ret[3] = bby2 - bby1
        return ret
    def to_tlwh(self, xyah):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = xyah.copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    def box_area(self, boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.
        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format
        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Return generalized intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            boxes1 (Tensor[N, 4])
            boxes2 (Tensor[M, 4])
        Returns:
            generalized_iou (Tensor[N, M]): the NxM matrix containing the pairwise generalized_IoU values
            for every element in boxes1 and boxes2
        """

        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union

        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        areai = whi[:, :, 0] * whi[:, :, 1]

        return iou - (areai - union) / areai
    def __getitem__(self, idx):
        '''
        for jj in range(len(self.framelist)):
            if self.framelist[jj] == 'PETS09-S2L1/000053':
                idx = jj
        '''
        '''
        for jj in range(len(self.framelist)):
            if self.framelist[jj] == 'MOT17-13/000451':
                idx = jj
        '''
        # print(self.framelist[idx])
        video_name = self.framelist[idx].split('/')[0]
        warp_mat = np.load(os.path.join("warp_mat", "%s.npy" %video_name))
        current_frame = int(self.framelist[idx].split('/')[-1])
        gt_path = cfg.DATA.PATH_TO_DATA_DIR + video_name + '/gt/gt.txt'
        # data/MOT/train/PETS09-S2L1/gt/gt.txt
        # data/MOT/val/MOT17-04/gt/gt.txt
        #######################################################
        ### load all detection info/feat in video
        sequence_dir = cfg.DATA.PATH_TO_DATA_DIR + video_name
        npy_file = cfg.DATA.PATH_TO_NPY_DIR + video_name + '-gt.npy'
        seq = self.gather_sequence_info(sequence_dir, npy_file)
        detections = seq["detections"]
        ### detection feat in current frame
        frame_indices = detections[:, 0].astype(np.int)
        mask = frame_indices == current_frame
        geo_current_frame = []
        feat_current_frame = []
        det_id_all = []
        # print(seq["image_size"][0],seq["image_size"][1])
        #print(len(detections[mask]),self.framelist[idx])
        for row in detections[mask]:
            if video_name in cfg.DATA.MOT17ALLNAME:
                if int(row[6]) != 0 and (int(row[7]) == 1 or int(row[7]) == 2) and float(row[8])>=0.2:
                    xyah = self.to_xyah(row[2:6],seq["image_size"][1],seq["image_size"][0])
                    tlwh0 = self.to_tlwh(xyah)
                    # geo_current_frame.append([xyah[0]/seq["image_size"][1],xyah[1]/seq["image_size"][0],xyah[2],xyah[3]/seq["image_size"][0]])
                    geo_current_frame.append([tlwh0[0],tlwh0[1],tlwh0[2]+tlwh0[0],tlwh0[3]+tlwh0[1]])
                    # print(row[2:6])
                    feat_current_frame.append(row[9:])
                    det_id_all.append(row[1])
            elif video_name in cfg.DATA.MOT15NAME:
                if int(row[6]) != 0:
                    xyah = self.to_xyah(row[2:6],seq["image_size"][1],seq["image_size"][0])
                    tlwh0 = self.to_tlwh(xyah)
                    # geo_current_frame.append([xyah[0]/seq["image_size"][1],xyah[1]/seq["image_size"][0],xyah[2],xyah[3]/seq["image_size"][0]])
                    geo_current_frame.append([tlwh0[0],tlwh0[1],tlwh0[2]+tlwh0[0],tlwh0[3]+tlwh0[1]])
                    feat_current_frame.append(row[10:])
                    det_id_all.append(row[1])
            else:
                raise RuntimeError('video name error!')
        if len(det_id_all) == 1:
            idx = random.randint(0, len(self.framelist) - 1)
            dic = self.__getitem__(idx)
            print('new!!!!!!!!!!!!!!!!!!!!!!!',self.framelist[idx])
            return dic
        ### tracklet features 
        mask_tracklet = frame_indices < current_frame
        dets_past = detections[mask_tracklet]
        delete_index = []
        if video_name in cfg.DATA.MOT17ALLNAME:
            for i in range(dets_past.shape[0]):
                if int(dets_past[i, 6]) == 0 or (int(dets_past[i, 7]) != 1 and int(dets_past[i, 7]) != 2) or float(dets_past[i, 8])<0.2:
                    delete_index.append(i)
            dets_past = np.delete(dets_past, delete_index, 0)
        elif video_name in cfg.DATA.MOT15NAME:
            for i in range(dets_past.shape[0]):
                if int(dets_past[i, 6]) == 0:
                    delete_index.append(i)
            dets_past = np.delete(dets_past, delete_index, 0)
        else:
            raise RuntimeError('video name error!')
        tracklet_id = dets_past[:, 1].astype(np.int)
        max_tracklet_id = tracklet_id.max()
        tracklet_feat_all = []
        tracklet_id_all = []
        tracklet_geometric_all = []
        tracklet_feat2_all = []
        for i in range(1, max_tracklet_id+1):
            tracklet_feat2 = []
            tracklet_id_mask = tracklet_id == i
            if (tracklet_id_mask==False).all():
                continue
            max_frame_in_this_tracklet = dets_past[tracklet_id_mask][:,0].astype(np.int).max()
            if max_frame_in_this_tracklet < current_frame - cfg.DATA.MAXAGE:
                continue
            tracklet_i = dets_past[tracklet_id_mask]
            saved_feat = np.array([])
            if video_name in cfg.DATA.MOT17ALLNAME:
                # tracklet_feat = tracklet_i[tracklet_i.shape[0]-1,9:]
                for j in range(tracklet_i.shape[0]):
                    tracklet_feat2.append(tracklet_i[j,9:])
                    tracklet_feat = self.moving_average(tracklet_i[j,9:], saved_feat, cfg.TRAIN.MOVING_AVERAGE_ALPHA)
                    saved_feat = tracklet_feat
                # if len(tracklet_feat2) > 20:
                #     tracklet_feat2 = tracklet_feat2[-20:]
                '''
                for j in range(tracklet_i.shape[0]):
                    tracklet_feat = tracklet_i[j,9:]
                '''
            else:
                for j in range(tracklet_i.shape[0]):
                    tracklet_feat2.append(tracklet_i[j,10:])
                    tracklet_feat = self.moving_average(tracklet_i[j,10:], saved_feat, cfg.TRAIN.MOVING_AVERAGE_ALPHA)
                    saved_feat = tracklet_feat
                # if len(tracklet_feat2) > 20:
                #     tracklet_feat2 = tracklet_feat2[-20:]
            ########## ECC + Kalman Filter #############
            # print(int(tracklet_i[:,0].min()), int(tracklet_i[:,0].max()))
            # print(tracklet_i[:,0])
            for tt in range(int(tracklet_i[:,0].min()), int(tracklet_i[:,0].max()) + 1):
                if tt == int(tracklet_i[:,0].min()):
                    tlwh = tracklet_i[0,2:6]
                    # print(tlwh, tracklet_i[j,:9])
                    mean, var = self.kf.initiate(self.to_xyah(tlwh,seq["image_size"][1],seq["image_size"][0]))
                else:
                    mean, var = self.kf.predict(mean, var, warp_mat[tt-2])
                    if tt in tracklet_i[:,0]:
                        tlwh_det = tracklet_i[tracklet_i[:,0] == tt, 2:6][0]
                        mean, var = self.kf.update(mean, var, self.to_xyah(tlwh_det,seq["image_size"][1],seq["image_size"][0]))
            for ii in range(int(tracklet_i[:,0].max())+1,current_frame+1):
                mean, var = self.kf.predict(mean, var, warp_mat[ii-2])
            
            ########## ECC + Kalman Filter #############
            # print(mean[:4].shape)
            tracklet_feat_all.append(tracklet_feat)
            tracklet_feat2_all.append(tracklet_feat2)
            # tracklet_geometric_all.append([mean[0]/seq["image_size"][1],mean[1]/seq["image_size"][0],mean[2],mean[3]/seq["image_size"][0]])
            mean_tlwh = self.to_tlwh(mean[:4])
            tracklet_geometric_all.append([mean_tlwh[0],mean_tlwh[1],mean_tlwh[2]+mean_tlwh[0],mean_tlwh[3]+mean_tlwh[1]])
            tracklet_id_all.append(int(tracklet_i[0, 1]))
        ###############################################
        # print(tracklet_id_all)
        # print(det_id_all)
        # print(len(tracklet_id_all),len(tracklet_feat_all),len(tracklet_geometric_all),len(det_id_all),len(geo_current_frame),len(feat_current_frame))
        
        noneed_detid = []
        for i in range(len(det_id_all)):
            if det_id_all[i] not in tracklet_id_all:
                noneed_detid.append(i)
        det_id_all = np.delete(np.array(det_id_all), noneed_detid, 0)
        geo_current_frame = np.delete(np.array(geo_current_frame), noneed_detid, 0)
        feat_current_frame = np.delete(np.array(feat_current_frame), noneed_detid, 0)
        if len(det_id_all) <= 1:
            idx = random.randint(0, len(self.framelist) - 1)
            dic = self.__getitem__(idx)
            print('new!!!!!!!!!!!!!!!!!!!!!!!',self.framelist[idx])
            return dic
        
        ########################################################
        # TODO: augmentation
        #print('det')
        #print(video_name, current_frame)
        graph_dets = self.construct_graph(feat_current_frame, geo_current_frame, [])
        #print('track')
        graph_tracks = self.construct_graph(tracklet_feat_all, tracklet_geometric_all, tracklet_feat2_all)
        # annotation permutation matrix [n_dets * n_tras] 
        n_dets = len(det_id_all)
        n_tras = len(tracklet_id_all)
        anno = torch.zeros(n_tras,n_dets)
        for i in range(n_tras):
            for j in range(n_dets):
                if tracklet_id_all[i] == det_id_all[j]:
                    anno[i][j] = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda:"+re.split(r",",cfg.gpu_id)[0] if torch.cuda.is_available() else "cpu")
        iou = torchvision.ops.box_iou(graph_tracks.geo,graph_dets.geo)
        mot_dict = {
            "graph_tracks": graph_tracks.to(device),
            "graph_dets": graph_dets.to(device),
            "anno": anno.to(device),
            "iou": iou.to(device)
        }
        
        # print(iou)
        # print(mot_dict["graph_tracks"].geo)
        # print(mot_dict["graph_dets"].geo)
        del graph_tracks
        del graph_dets
        del anno
        del tracklet_feat_all
        del detections
        del seq
        del feat_current_frame
        del dets_past
        del tracklet_feat
        gc.collect()
        
        #print(mot_dict["graph_tracks"].x[0][0])
        return mot_dict