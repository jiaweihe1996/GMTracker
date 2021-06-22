import os 
import subprocess
from tqdm import tqdm
from utils.config import cfg


for i, seq in tqdm(enumerate(os.listdir(cfg.DATA.VAL_DIR))):
    if seq in cfg.DATA.MOVING:
        if seq in cfg.DATA.FOLD0_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.moving_checkpoint_dir,'fold0'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)
        if seq in cfg.DATA.FOLD1_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.moving_checkpoint_dir,'fold1'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)
        if seq in cfg.DATA.FOLD2_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.moving_checkpoint_dir,'fold2'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)

    if seq in cfg.DATA.STATIC:
        if seq in cfg.DATA.FOLD0_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.static_checkpoint_dir,'fold0'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)
        if seq in cfg.DATA.FOLD1_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.static_checkpoint_dir,'fold1'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)
        if seq in cfg.DATA.FOLD2_VAL:
            for detection in cfg.detections:
                process = ['python', 'gmtracker_app.py', '--sequence_dir='+os.path.join(cfg.DATA.VAL_DIR,seq), '--detection_file=npy/npyval_tracktor/'+seq+'-'+detection+'.npy', '--checkpoint_dir='+os.path.join(cfg.static_checkpoint_dir,'fold2'), '--max_age=100', '--reid_thr=0.7', '--output_file='+os.path.join(cfg.RESULTS_VAL_DIR,seq+'-'+detection+'.txt')]
                print(process)
                subprocess.call(process)
