
import os 
import subprocess
from tqdm import tqdm
from utils.config import cfg

for i, seq in tqdm(enumerate(os.listdir(cfg.DATA.TEST_DIR))):
    if seq in cfg.DATA.MOVING:
        for detection in cfg.detections:
            process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.TEST_DIR, seq), '--detection_file=npy/npytest_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+cfg.moving_checkpoint_dir, '--max_age=100', '--reid_thr=0.6', '--output_file='+os.path.join(cfg.RESULTS_DIR,seq+'-'+detection+'.txt')]
            print(process)
            subprocess.call(process)
    if seq in cfg.DATA.STATIC:
        for detection in cfg.detections:
            process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.TEST_DIR, seq), '--detection_file=npy/npytest_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+cfg.static_checkpoint_dir, '--max_age=100', '--reid_thr=0.6', '--output_file='+os.path.join(cfg.RESULTS_DIR, seq+'-'+detection+'.txt')]
            print(process)
            subprocess.call(process)