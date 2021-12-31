import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from qpth.qp import QPFunction
import numpy as np
from GMMOT.graph import ReidEncoder
from utils.build_graphs import reshape_edge_feature
from utils.voting_layer import Voting
from GMMOT.config import cfg
import gc
def gh(n):

    A = np.ones((n, n)) - np.eye(n)

    edge_num = int(np.sum(A, axis=(0, 1)))

    n_pad = n
    edge_pad = edge_num

    G = np.zeros((n_pad, edge_pad), dtype=np.float32)
    H = np.zeros((n_pad, edge_pad), dtype=np.float32)
    start = np.zeros((n_pad, n_pad), dtype=np.float32)
    end = np.zeros((n_pad, n_pad), dtype=np.float32)
    edge_idx = 0
    for i in range(n):
        for j in range(n):
            start[i,j] = i
            end[i,j] = j
            if A[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1

    return G, H, start,end

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB
def moving_average(feat, saved_ma, alpha):
    if len(saved_ma) == 0:
        ema = feat
    else:
        ema = saved_ma * alpha + feat * (1 - alpha)
    return ema
def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

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
class GMNet(nn.Module):
    def __init__(self):
        super(GMNet, self).__init__()
        self.reid_enc = ReidEncoder()
        self.voting_layer = Voting(alpha=cfg.TRAIN.VOTING_ALPHA)
        self.cross_graph = nn.Linear(512, 512)
        # self.cross_graph2 = nn.Linear(512, 512)
        # nn.init.normal_(self.cross_graph.weight, 0, 0.001)
        nn.init.eye_(self.cross_graph.weight)
        nn.init.constant_(self.cross_graph.bias, 0)

    def forward(self, tra, det,iou):
        feat_tra_all = []
        # feat_tra = self.reid_enc(tra.x)
        for tra_i in range(len(tra.fnm)):
            # print(tra.fnm[tra_i])
            # feat_tra0_o = F.normalize(tra.fnm[0][tra_i], p=2, dim=1)   
            feat_tra0 = self.reid_enc(tra.fnm[tra_i])
            tracklet_feat = feat_tra0.mean(0).unsqueeze(0)
            feat_tra_all.append(tracklet_feat)

        feat_tra = torch.cat(feat_tra_all)
        # print(feat_tra.shape)        
        feat_det = self.reid_enc(det.x)

        data1 = feat_tra
        data2 = feat_det
        # data1 = F.normalize(feat_tra, p=2, dim=1)
        # print(data1.shape)
        # data2 = F.normalize(feat_det, p=2, dim=1)
        # print('data1',data1[0])
        # print('data2',data2[0])
        #print(data1.shape)
        G1, H1, start_src, end_src = gh(data1.shape[0])
        G2, H2, start_tgt, end_tgt = gh(data2.shape[0])
        start_src = torch.tensor(start_src)
        end_src = torch.tensor(end_src)
        start_tgt = torch.tensor(start_tgt)
        end_tgt = torch.tensor(end_tgt)
        G_src = torch.tensor(G1).unsqueeze(0)
        G_tgt = torch.tensor(G2).unsqueeze(0)
        H_src = torch.tensor(H1).unsqueeze(0)
        H_tgt = torch.tensor(H2).unsqueeze(0)
        U_src = data1.t().unsqueeze(0).cpu()
        U_tgt = data2.t().unsqueeze(0).cpu()

        Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt).cuda()
        
        Mp0 = Mp0+iou
        # print(Mp0,Mp0.shape)
        emb1, emb2 = U_src.transpose(1, 2).cuda(), U_tgt.transpose(1, 2).cuda()

        
        m_emb1 = torch.bmm(Mp0, emb2)
        m_emb2 = torch.bmm(Mp0.transpose(1, 2), emb1)
        # print(torch.norm(emb1,p=2,dim=2,keepdim=True).repeat(1,1,512).shape,torch.norm(m_emb1,p=2,dim=2,keepdim=True).repeat(1,1,512).shape)
        # print(torch.norm(emb2,p=2,dim=2,keepdim=True).repeat(1,1,512).shape, torch.norm(m_emb2,p=2,dim=2,keepdim=True).repeat(1,1,512).shape)
        lambda_1 = (torch.norm(emb1,p=2,dim=2,keepdim=True).repeat(1,1,512) / torch.norm(m_emb1,p=2,dim=2,keepdim=True).repeat(1,1,512)).detach()
        lambda_2 = (torch.norm(emb2,p=2,dim=2,keepdim=True).repeat(1,1,512) / torch.norm(m_emb2,p=2,dim=2,keepdim=True).repeat(1,1,512)).detach()
        # emb1_new0 = F.normalize((emb1+m_emb1).squeeze(0), p=2, dim=1).unsqueeze(0)
        # emb2_new0 = F.normalize((emb2+m_emb2).squeeze(0), p=2, dim=1).unsqueeze(0)
        emb1_new = F.relu(self.cross_graph(emb1+lambda_1*m_emb1))
        emb2_new = F.relu(self.cross_graph(emb2+lambda_2*m_emb2))
        # emb1_new = F.relu(self.cross_graph2(emb1_new))
        # emb2_new = F.relu(self.cross_graph2(emb2_new))

        emb1_new = F.normalize(emb1_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        emb2_new = F.normalize(emb2_new.squeeze(0), p=2, dim=1).unsqueeze(0)

        Mp = torch.matmul(emb1_new, emb2_new.transpose(1, 2)).squeeze(0)


        kro_one_src = torch.ones(emb1_new.shape[1],emb1_new.shape[1])
        kro_one_tgt = torch.ones(emb2_new.shape[1],emb2_new.shape[1])
        mee1 = kronecker(kro_one_tgt,start_src).long()
        mee2 = kronecker(kro_one_tgt,end_src).long()
        mee3 = kronecker(start_tgt,kro_one_src).long()
        mee4 = kronecker(end_tgt,kro_one_src).long()
        src = torch.cat([emb1_new.squeeze(0).unsqueeze(1).repeat(1,emb1_new.shape[1],1),emb1_new.repeat(emb1_new.shape[1],1,1)],dim=2)
        tgt = torch.cat([emb2_new.squeeze(0).unsqueeze(1).repeat(1,emb2_new.shape[1],1),emb2_new.repeat(emb2_new.shape[1],1,1)],dim=2)
        src_tgt = (src.reshape(-1,1024)@tgt.reshape(-1,1024).t()).reshape(emb1_new.shape[1],emb1_new.shape[1],emb2_new.shape[1],emb2_new.shape[1])
        mask = ((mee1-mee2).bool()&(mee3-mee4).bool()).float().cuda()
        M = src_tgt[mee1,mee2,mee3,mee4]/2
        M = mask*M
        M = M.unsqueeze(0)
        k = (Mp.shape[0]-1)*(Mp.shape[1]-1)
        M[0] = k*torch.eye(M.shape[1],M.shape[2]).cuda() - M[0]
        M = M.squeeze(0)
        Mpp = Mp.transpose(0, 1).reshape(Mp.shape[0]*Mp.shape[1]).cuda()










        # X2 = reshape_edge_feature(emb1_new.transpose(1, 2).cpu(), G_src, H_src)
        # Y2 = reshape_edge_feature(emb2_new.transpose(1, 2).cpu(), G_tgt, H_tgt)

        # Me = torch.matmul(X2.transpose(1, 2), Y2).squeeze(0)/2
        # # Mp = torch.matmul(U_src.transpose(1, 2), U_tgt).squeeze(0).detach()
        # Mp = torch.matmul(emb1_new, emb2_new.transpose(1, 2)).squeeze(0)
        # # print(Mp.shape)

        # #print(Mp.shape)
        # a1 = Me.transpose(0, 1)
        # a2 = a1.reshape(Me.shape[0]*Me.shape[1])
        # K_G = kronecker(G_tgt.squeeze(0),G_src.squeeze(0)).detach()
        # #print(K_G.shape)
        # K1Me = a2*K_G
        # del K_G
        # del a2
        # del Me
        # gc.collect()
        # torch.cuda.empty_cache()
        # K_H = kronecker(H_tgt.squeeze(0),H_src.squeeze(0)).detach()
        # M = torch.mm(K1Me,K_H.t())
        # del K1Me
        # del K_H
        # gc.collect()
        # torch.cuda.empty_cache()
        # Mpp = Mp.transpose(0, 1).reshape(Mp.shape[0]*Mp.shape[1]).cuda()
        # M = M.unsqueeze(0).cuda()
        # k = (Mp.shape[0]-1)*(Mp.shape[1]-1)
        # #print(k)
        # M[0] = k*torch.eye(M.shape[1],M.shape[2]).cuda() - M[0]
        # #print(M[0])
        # #M[0] = torch.cholesky(M[0])
        # M = M.squeeze(0)

        #cvxpy/expression 516 warning
        if Mp.shape[0] > Mp.shape[1]:
            n, m, p = M.shape[0], Mp.shape[1], Mp.shape[0]
            a = torch.zeros(p, n).cuda()
            b = torch.zeros(m, n).cuda()
            for i in range(p):
                for j in range(m):
                    a[i][j*p+i]=1
            for i in range(m):
                b[i][i*p:(i+1)*p]=1

            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n).cuda()
            h = torch.zeros(n).cuda()
            bb = torch.ones(m).cuda()
            bbb = torch.ones(p).cuda()
            hh = torch.cat((h,bbb))
            GG = torch.cat((G,a),0)
            s = qp(M,-Mpp,GG,hh,b,bb)

            s = s.reshape(Mp.shape[1], Mp.shape[0]).unsqueeze(0)
            s = torch.relu(s)-torch.relu(s-1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach())).permute(0,2,1)
            s = self.voting_layer(s, torch.tensor([Mp.shape[1]]), torch.tensor([Mp.shape[0]])).permute(0,2,1)
            #print(s)
        elif Mp.shape[0] == Mp.shape[1]:
            n, m, p = M.shape[0], Mp.shape[0], Mp.shape[1]
            a = torch.zeros(m+p, n).cuda()
            for i in range(p):
                for j in range(m):
                    a[i][j*p+i]=1
            for i in range(m):
                a[i+p][i*p:(i+1)*p]=1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n).cuda()
            h = torch.zeros(n).cuda()
            b = torch.ones(m+p).cuda()
            s = qp(M,-Mpp,G,h,a,b)
            
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s)-torch.relu(s-1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach()))
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))
            #print(s)
        else:
            n, m, p = M.shape[0], Mp.shape[0], Mp.shape[1]
            a = torch.zeros(p, n).cuda()
            b = torch.zeros(m, n).cuda()
            for i in range(p):
                for j in range(m):
                    a[i][j*p+i]=1
            for i in range(m):
                b[i][i*p:(i+1)*p]=1
            qp = QPFunction(check_Q_spd=False)
            G = -torch.eye(n).cuda()
            h = torch.zeros(n).cuda()
            bb = torch.ones(m).cuda()
            bbb = torch.ones(p).cuda()
            hh = torch.cat((h,bbb))
            GG = torch.cat((G,a),0)
            s = qp(M,-Mpp,GG,hh,b,bb)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s)-torch.relu(s-1)
            # s = s-s.min().detach()
            # s = torch.sigmoid((s-s.mean().detach())/torch.sqrt(s.var().detach()))
            s = self.voting_layer(s, torch.tensor([Mp.shape[0]]), torch.tensor([Mp.shape[1]]))

        return s



if __name__ == "__main__":
    GMNet()
    print('aaa')
