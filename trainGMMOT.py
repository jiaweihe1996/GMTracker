import torch
import torch.optim as optim
import numpy as np
from GMMOT.dataloader import MOTGraph
from GMMOT.model import GMNet
import torch.nn as nn
from GMMOT.config import cfg
from torch_geometric.data import DataLoader
from pathlib import Path
import os
import scipy
import scipy.optimize as opt
from torch.optim.lr_scheduler import CosineAnnealingLR

class Loss(nn.Module):
    def forward(self, x, gt):
        
        x = x.reshape(-1)
        gt = gt.reshape(-1)
        
        pa = len(gt)*len(gt)/(len(gt)-gt.sum())/2/gt.sum()
        weight_0 = pa*gt.sum()/len(gt)
        weight_1 = pa*(len(gt) - gt.sum())/len(gt)
        weight = torch.zeros(len(gt))
        for i in range(len(gt)):
            if gt[i] == 0:
                weight[i] = weight_0
            elif gt[i] == 1:
                weight[i] = weight_1
            else:
                raise RuntimeError('loss weight error')
        #print(weight)
        #weight = torch.abs(gt-0.1).detach()
        loss = nn.BCELoss(weight=weight.cuda())
        
        # loss = nn.BCELoss(reduction='sum')
        # print(gt.sum())
        out = loss(x, gt)
        return out

def cal_acc(x, gt):
    x = x.squeeze(0)
    #print(gt.shape)
    gt = gt.squeeze(0)
    if x.shape[0]>x.shape[1]:
        x = x.t()
        gt = gt.t()
    
    b = x.max(1).indices == gt.max(1).indices
    if b.shape[0] != x.shape[0]:
        raise RuntimeError('b.shape[0] != x.shape[0]')
    return int(b.sum()), b.shape[0]
def train_model(model, criterion, optimizer, dataloader, num_epochs,scheduler):
    print("Start training...")

    dataset_size = len(dataloader.dataset)

 
    if cfg.resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path),strict=False)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        model.train()  # Set model to training mode

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        iter_num = 0
        tp_sum = 0.0
        node = 0.0
        epoch_iter = len(dataloader)
        # Iterate over data.
        for inputs in dataloader:
            iter_num = iter_num + 1
            graph_tracks = inputs["graph_tracks"]
            graph_dets = inputs["graph_dets"]
            anno = inputs["anno"]
            iou = inputs["iou"]
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                
                # TODO: model
                s_pred = model(graph_tracks, graph_dets,iou)
                #print(perm_mat_list[0][0])
                # print(s_pred.shape,anno.shape)
                loss = criterion(s_pred, anno)

                #print(s_pred,anno)
                tp, n_node = cal_acc(s_pred, anno)
                #print(fp, n_node)
                # backward + optimize
                loss.backward()
                # print(model.reid_enc.fc1.weight.grad)	
                if torch.isnan(model.reid_enc.fc1.weight.grad).max():
                    continue

                optimizer.step()
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                accdict = dict()
                accdict['matching accuracy'] = tp/n_node
                
                torch.cuda.empty_cache()
                # statistics
                bs = cfg.TRAIN.BATCH_SIZE
                running_loss += loss.item() * bs  # multiply with batch size
                epoch_loss += loss.item() * bs
                tp_sum += tp
                node += n_node

                if iter_num % cfg.STATISTIC_STEP == 0:
                    loss_avg = running_loss / cfg.STATISTIC_STEP / bs
                    acc_avg = tp_sum / node
                    print(
                        "Epoch {:<4} Iter {:<4} Loss={:<8.4f} Acc={:<8.4f}tp_sum {:<4}node {:<4}".format(
                            epoch, iter_num, loss_avg, acc_avg,tp_sum,node
                        )
                    )
                    running_loss = 0.0
                    tp_sum = 0.0
                    node = 0.0
                    

        #epoch_loss = epoch_loss / dataset_size
        
        if cfg.save_checkpoint:
            checkpoint_path = Path(cfg.model_dir) / "params"
            if not checkpoint_path.exists():
                checkpoint_path.mkdir(parents=True)
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))
        

        print()
        scheduler.step()

    return model


if __name__ == "__main__":

    torch.manual_seed(cfg.RANDOM_SEED)
    dataset = MOTGraph()
    dataloader =  DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True
    )
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GMNet()
    model = model.cuda()
    #model = torch_geometric.nn.DataParallel(model,device_ids=[0,1])
    

    criterion = Loss()

    params = [param for param in model.parameters()]
    optimizer = optim.Adam(params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    num_epochs = cfg.TRAIN.MAXEPOCH

    model = train_model(
        model,
        criterion,
        optimizer,
        dataloader,
        num_epochs,
        scheduler
    )
