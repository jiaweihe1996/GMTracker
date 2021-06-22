

# GMTracker

This repository is the official PyTorch implementation of the CVPR 2021 paper: *Learnable Graph Matching: Incorporating Graph Partitioning with Deep Feature Learning for Multiple Object Tracking*.   
[[arXiv](https://arxiv.org/abs/2103.16178)] [[CVF open access](https://openaccess.thecvf.com/content/CVPR2021/html/He_Learnable_Graph_Matching_Incorporating_Graph_Partitioning_With_Deep_Feature_Learning_CVPR_2021_paper.html)]

- **Training and pre-processing code will be released in next few days.**

![Method Visualization](vis/pipeline.jpg)

## Getting Started

1. Clone this repository:
```clone
git clone --recursive https://github.com/jiaweihe1996/GMTracker
```
2. Install requirements:
- Python == 3.6.X
- PyTorch >= 1.4 with CUDA >=10.0 (tested on PyTorch 1.4.0)
- torchvision 

```setup
pip install -r requirements.txt
# Install scs-gpu
pip uninstall scs
cd scs-python
python setup.py install --scs --gpu
```
3. Download the [MOT17](https://motchallenge.net/data/MOT17.zip) data and unzip. The data files' structure is like
```
--- data  
    --- MOT17 
        --- train  
            --- MOT17-02  
            --- MOT17-04  
            ...  
        --- test  
            --- MOT17-01  
            --- MOT17-03  
            ...  
```
4. Extract inital ReID features:

- **(Preference)** For convenience, we provide the preprocessed detection appearance features, which are stored in `npy` files. You can download them from [GoogleDrive](https://drive.google.com/file/d/1POVU2mWBet6QVX-hOoexeecNU0KrNl6c/view?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1SOL1hAIrSzYBRsyMYKyOIw) (code: dyvk) and unzip it.
- Or get refined detections and extract inital ReID features from the ReID model (Will be released later).


5. Run GMTracker on a sequence:
```demo
python gmtracker_app.py --sequence_dir /path/to/MOT/sequence --detection_file /path/to/detection.npy  --checkpoint_dir /path/to/checkpoint_dir --max_age 100 --reid_thr 0.6 --output_file /path/to/output.txt
```
For example, on MOT17-01 sequence (static camera) with DPM detector:
```
python gmtracker_app.py --sequence_dir data/MOT17/test/MOT17-01 --detection_file npy/npytest_tracktor/MOT17-01-DPM.npy  --checkpoint_dir experiments/static/params/0001 --max_age 100 --reid_thr 0.6 --output_file results/test/MOT17-01-DPM.txt
```
or cross validation on MOT17-05-DPM (moving camera, fold2 in val set):
```
python gmtracker_app.py --sequence_dir data/MOT17/train/MOT17-05 --detection_file npy/npyval_tracktor/MOT17-05-DPM.npy  --checkpoint_dir experiments/moving/params/0001/fold2 --max_age 100 --reid_thr 0.6 --output_file results/crossval/MOT17-05-DPM.txt
```
 - attributes of each sequences:
 ```
FOLD0_VAL = ['MOT17-02', 'MOT17-10', 'MOT17-13']
FOLD1_VAL = ['MOT17-04', 'MOT17-11']
FOLD2_VAL = ['MOT17-05', 'MOT17-09']

STATIC = ['MOT17-01', 'MOT17-03', 'MOT17-08', 'MOT17-02', 'MOT17-04', 'MOT17-09']
MOVING = ['MOT17-06', 'MOT17-07', 'MOT17-12', 'MOT17-14', 'MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13']
 ```
6. Track on all sequences on MOT17 test set:
```all
python motchallenge_tracking.py
```
7. Visualize tracking results:
```demo
python show_results.py --sequence_dir /path/to/MOT/sequence --result_file /path/to/result.txt --output_file /path/to/output.avi
```
8. Cross validation for all sequences on MOT17:
```val
python cross_validation.py
```
9. Evaluate cross validation results:
- You should first organize the validation data folder and put the groundtruth file at `MOT17/val/sequense-det/gt/gt.txt` like
```
--- val
    --- MOT17-02-DPM
        --- gt
            ---gt.txt
    --- MOT17-02-FRCNN
        ...
    --- MOT17-02-SDP
        ...
    --- MOT17-04-DPM
    ...
```
- and run:
```
python -m motmetrics.apps.eval_motchallenge ./MOT17/val ./result/val
```
## Acknowledgement

This implementation is mainly based on [deep_sort](https://github.com/nwojke/deep_sort) repo under GPL-3.0 License. Our ReID model is trained via [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) repo. The codes in qpth folder are mainly from [qpth](https://github.com/locuslab/qpth).
## Citing GMTracker

If you find this repo useful in your research, please consider citing the following paper:

```
@InProceedings{he2021gmtracker,
    author    = {He, Jiawei and Huang, Zehao and Wang, Naiyan and Zhang, Zhaoxiang},
    title     = {Learnable Graph Matching: Incorporating Graph Partitioning With Deep Feature Learning for Multiple Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {5299-5309}
}
```
