import torch.nn as nn
import torch
import numpy as np
import cvxpy as cp
import torch
import torch.nn.functional as F
from utils.config import cfg
from GMMOT.graph_encoder import ReidEncoder

class GraphNet(nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.reid_enc = ReidEncoder()
        self.cross_graph = nn.Linear(512, 512)
    def kronecker(self,A, B):
        AB = torch.einsum("ab,cd->acbd", A, B)
        AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
        return AB
    def forward(self, U_src, U_tgt, kf_gate,reid_thr,iou,start_src,end_src,start_tgt,end_tgt,seq_name,inverse_flag):

        # different aggregation weights in sequences taken by static/moving camera
        if seq_name in cfg.DATA.STATIC:
            Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt) + iou.unsqueeze(0)
        elif seq_name in cfg.DATA.MOVING:
            Mp0 = torch.matmul(U_src.transpose(1, 2), U_tgt)

        emb1, emb2 = U_src.transpose(1, 2), U_tgt.transpose(1, 2)
        
        # Cross-graph GCN
        m_emb1 = torch.bmm(Mp0, emb2)
        m_emb2 = torch.bmm(Mp0.transpose(1, 2), emb1)
        lambda_1 = (torch.norm(emb1,p=2,dim=2,keepdim=True).repeat(1,1,512) / torch.norm(m_emb1,p=2,dim=2,keepdim=True).repeat(1,1,512))
        lambda_2 = (torch.norm(emb2,p=2,dim=2,keepdim=True).repeat(1,1,512) / torch.norm(m_emb2,p=2,dim=2,keepdim=True).repeat(1,1,512))
        emb1_new = F.relu(self.cross_graph(emb1+lambda_1*m_emb1))
        emb2_new = F.relu(self.cross_graph(emb2+lambda_2*m_emb2))

        emb1_new = F.normalize(emb1_new.squeeze(0), p=2, dim=1).unsqueeze(0)
        emb2_new = F.normalize(emb2_new.squeeze(0), p=2, dim=1).unsqueeze(0)
       
        # calculate the vertex-vertex similarity and edge-edge similarity
        Mp = torch.matmul(emb1_new, emb2_new.transpose(1, 2)).squeeze(0)
        Mpp = Mp.transpose(0, 1).reshape(Mp.shape[0]*Mp.shape[1]).unsqueeze(0).t()
        if Mp.shape[0]==1 and Mp.shape[1]==1:
            thr_flag = torch.Tensor(Mp.shape[0],Mp.shape[1]).zero_()
            for i in range(Mp.shape[0]):
                for j in range(Mp.shape[1]):
                    if kf_gate[i][j] == -1 or iou[i][j]==0 or Mp[i][j]<reid_thr:
                        thr_flag[i][j] = 1
            return np.array([0,0]),thr_flag

        
        kro_one_src = torch.ones(emb1_new.shape[1],emb1_new.shape[1])
        kro_one_tgt = torch.ones(emb2_new.shape[1],emb2_new.shape[1])
        mee1 = self.kronecker(kro_one_tgt,start_src).long()
        mee2 = self.kronecker(kro_one_tgt,end_src).long()
        mee3 = self.kronecker(start_tgt,kro_one_src).long()
        mee4 = self.kronecker(end_tgt,kro_one_src).long()
        src = torch.cat([emb1_new.squeeze(0).unsqueeze(1).repeat(1,emb1_new.shape[1],1),emb1_new.repeat(emb1_new.shape[1],1,1)],dim=2)
        tgt = torch.cat([emb2_new.squeeze(0).unsqueeze(1).repeat(1,emb2_new.shape[1],1),emb2_new.repeat(emb2_new.shape[1],1,1)],dim=2)
        src_tgt = (src.reshape(-1,1024)@tgt.reshape(-1,1024).t()).reshape(emb1_new.shape[1],emb1_new.shape[1],emb2_new.shape[1],emb2_new.shape[1])
        mask = ((mee1-mee2).bool()&(mee3-mee4).bool()).float()
        M = src_tgt[mee1,mee2,mee3,mee4]/2
        M = mask*M
        M = M.unsqueeze(0)
        k = (Mp.shape[0]-1)*(Mp.shape[1]-1)
        M[0] = k*torch.eye(M.shape[1],M.shape[2]) - M[0]
        if Mp.shape[0]==1 or Mp.shape[1]==1:
            M[0] = torch.zeros_like(M[0])
            print('single')
        else:
            M[0] = torch.cholesky(M[0])

        # solve relaxed quadratic programming
        if Mp.shape[0] > Mp.shape[1]:
            n, m, p = M.shape[1], Mp.shape[1], Mp.shape[0]
            a = np.zeros((p, n))
            b = np.zeros((m, n))
            for i in range(p):
                for j in range(m):
                    a[i][j*p+i]=1
            for i in range(m):
                b[i][i*p:(i+1)*p]=1
            x = cp.Variable(n)
            obj = cp.Minimize(0.5*cp.sum_squares(M.squeeze(0).numpy()@x)-Mpp.numpy().T@x)
            cons = [a@x<=1,b@x==1, x>=0]
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.SCS, gpu=True, use_indirect=True)
            s = torch.tensor(x.value)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s)-torch.relu(s-1)
        elif Mp.shape[0] == Mp.shape[1]:
            n, m, p = M.shape[1], Mp.shape[0], Mp.shape[1]
            x = cp.Variable(n)
            a = np.zeros((m+p, n))
            for i in range(p):
                for j in range(m):
                    a[i][j*p+i]=1
            for i in range(m):
                a[i+p][i*p:(i+1)*p]=1
            obj = cp.Minimize(0.5*cp.sum_squares(M.squeeze(0).numpy()@x)-Mpp.numpy().T@x)
            cons = [a@x==1, x>=0]
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.SCS, gpu=True, use_indirect=True)
            s = torch.tensor(x.value)
            s = s.reshape(Mp.shape[1], Mp.shape[0]).t().unsqueeze(0)
            s = torch.relu(s)-torch.relu(s-1)

        # thresholds
        thr_flag = torch.Tensor(Mp.shape[0],Mp.shape[1]).zero_()
        for i in range(Mp.shape[0]):
            for j in range(Mp.shape[1]):
                if kf_gate[i][j] == -1 or iou[i][j]==0 or Mp[i][j]<reid_thr:
                    thr_flag[i][j] = 1

        # greedy matching from matching score map
        if s.shape[1] >= s.shape[2]:
            s = s.squeeze(0).t()
            s = np.array(s)
            n = min(s.shape)
            Y = s.copy()
            Z = np.zeros(Y.shape)
            replace = np.min(Y) - 1
            for i in range(n):
                z = np.unravel_index(np.argmax(Y), Y.shape)
                Z[z] = 1
                Y[z[0], :] = replace
                Y[:, z[1]] = replace
            match_tra = np.argmax(Z, 1)  
            match_tra = torch.tensor(match_tra)
            if inverse_flag == False:
                output = np.array(torch.cat([match_tra.unsqueeze(0),torch.arange(len(match_tra)).unsqueeze(0)],0)).T
            if inverse_flag == True:
                thr_flag = thr_flag.t()
                output = np.array(torch.cat([torch.arange(len(match_tra)).unsqueeze(0),match_tra.unsqueeze(0)],0)).T

        return output, thr_flag