from pathlib import Path
from tqdm.notebook import tqdm
import multiprocessing
import warnings
import gc
import torch
import os
import time
import random
# warnings.filterwarnings('ignore')

# This is the script to run the MP-KD model in parallel;
# It runs MP-KD on all tasks in the dataset folder;
# Tune $parellel to control the degree of parallel according to your hardware load.

data_name = 'wiki'
script = "main.py"
source_divide = "en"
parellel = 16

def run_model(task):
    task_id, task_dir = task[0], str(task[1]).split('/')[-1]
    # if source_divide not in task_dir:
    #     return None
    print(f"Running {script} on {task_dir}")
    device = int(task_id % torch.cuda.device_count())
    time.sleep(random.randint(0, 60)) # sleep for a while to avoid CPU overload
    for pre_train in [1, 0]:
        for num_neighbors in [8, 16, 32, 64]:
            comment = f"python {script} --task {task_dir}  --device {device} --use_pretrain {pre_train} --num_neighbors {num_neighbors}"
            os.system(comment)
    print(f"Completed {script} on {task_dir}")
    gc.collect()

def run_all_models():
    tasks = [
        (i, task) for i, task in enumerate(Path(f"../data/{data_name}").iterdir()) if task.is_dir()
    ]
    with multiprocessing.Pool(parellel) as p:
        p.map(run_model, tasks)

run_all_models()
