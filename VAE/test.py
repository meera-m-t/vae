import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from VAE.train_config import ExperimentationConfig


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

        TestSet = settings.get_test_dataset()
        test_set = TestSet(**settings.test_set_kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )

        test_metrics = settings.get_test_metrics()
        test_metrics_file = None
        image_count = 0
        test_loss = 0

        loss = settings.get_loss()
        criterion = loss(**settings.loss_kwargs)

        with torch.no_grad():
            batch_metrics = []
            for j, (X) in enumerate(test_loader):
                X = X.float().cuda()
                test_loss += criterion(model, X)

                batch_preds = []
                # if len(test_metrics):
                #     metric_values = {}
                #     for metric_name, metric in test_metrics.items():
                #         metric_value = metric(torch.stack(batch_preds), targets_cpu)
                #         metric_values[metric_name] = metric_value

                #         print(
                #             f"{metric_name}(Batch({j})): {metric_value}",
                #             file=test_metrics_file,
                #             end=" | ",
                #         )
                #     batch_metrics.append(metric_values)

                #     print(file=test_metrics_file)

            print("Test loss: ", test_loss / len(test_loader))
            # if test_metrics_file:
            #     test_metrics_file.close()
            #     mean_metrics = {}
            #     for key in batch_metrics[0].keys():
            #         mean_metrics[key] = sum(d[key] for d in batch_metrics) / len(
            #             batch_metrics
            #         )

            #     with open(
            #         f"{experiment_dir}/avg_test_metrics_epoch-{epoch}.json"
            #         if epoch
            #         else f"{experiment_dir}/avg_test_metrics_best.json",
            #         "w",
            #     ) as json_file:
            #         json.dump(mean_metrics, json_file, indent=2)
