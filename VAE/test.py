import json
from VAE.datasets import Dataset_LHS  
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from VAE.train_config import ExperimentationConfig
from VAE.utils import SimpleLogger, make_dirs
import shutil
def test(experiment_dir: str, epoch: int = None):
    with open(experiment_dir + "/TrainConfig.json", "r") as json_file:
        settings_json = json.load(json_file)
        settings = ExperimentationConfig.parse_obj(settings_json)
        Model = settings.get_model()
        model = Model(**settings.model_kwargs)
        if torch.cuda.is_available():
            model = model.cuda()
        if epoch is None:
            model.load_state_dict(torch.load(experiment_dir + "/best_weights.pt"))
        else:
            model.load_state_dict(
                torch.load(experiment_dir + f"epoch_{epoch}_weights.pt")
            )
        model.eval()

        TestSet = settings.get_train_dataset()      
        test_set = TestSet(**settings.test_set_kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )

    save_dir = (
        Path(settings.save_dir).resolve()
        / f"{experiment_dir}/output"
    )

   
    if save_dir.exists() and save_dir.is_dir():
        shutil.rmtree(save_dir)
    if not save_dir.exists():
        make_dirs([save_dir])
    

        with torch.no_grad():
            if test_set.num_dimensions in {2, 3}:                
                all_ys = [] 
                for j, (X) in enumerate(test_loader):                 
                    Y = model(X.float().cuda())                                
                    all_ys.append(torch.squeeze((Y[0])))

                ys = torch.cat(all_ys)                                    
                new_dataset = Dataset_LHS(data=ys)
                new_dataset.plot_dist(
                            save_dir / "reconstructed-distribution.png"
                    )






