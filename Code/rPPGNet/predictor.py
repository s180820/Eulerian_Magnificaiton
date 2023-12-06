import torch
from torch.nn import functional as F
import torchvision
from tqdm import tqdm
import os
from _config import wandb_defaults, default_config
from dataloader import CustomDataset
import wandb
from utils import download_model, ReduceLROnPlateau
import numpy as np
from models import models
from loss_functions import Neg_Pearson, Neg_Pearson2
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime as dt


class Predictor:
    def __init__(
        self,
        project="Eulerian_Magnification",
        name=None,
        model=None,
        ex_model=None,
        est_model=None,
        frames=64,
        config=None,
        use_wandb=True,
        verbose=True,
        model_save_freq=None,
        **kwargs,
    ):
        """
        Class for training and testing rPPG predicton from fases

        Parameters
        ----------
        name : str, optional
            Name of the run, by default 'run_' + current time
        config : dict, optional
            Dictionary with configuration, by default the defualt_config from hparams.py
        use_wandb : bool, optional
            Whether to use wandb, by default True
        verbose : bool, optional
            Whether to print progress, by default True
        show_test_images : bool, optional
            Whether to show test images, by default False
        model_save_freq : int, optional
            How often to save the model, by default None
            None means only save the best model
        **kwargs : dict
            Additional arguments to config

        Config
        ------
        num_epochs : int
            Number of epochs to train
        dropout : float
            Dropout rate
        batchnorm : bool
            Whether to use batchnormalisation
        train_dataset_kwargs : dict
            Additional arguments to HotdogDataset for training
            i.e. train_dataset_kwargs = {"data_augmentation": False}
        test_dataset_kwargs : dict
            Additional arguments to HotdogDataset for testing
        optimizer : str
            Optimizer to use (from torch.optim)
        optimizer_kwargs : dict
            Additional arguments to optimizer
            e.g. optimizer_kwargs = {"lr": 0.01}
        scheduler : bool
            Whether to use a scheduler - will use ExponentialLR with gamma = 0.1
            Decrease will happen after 20 % of epochs
        """

        # set info
        self.name = (
            name
            if name is not None
            else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_freq = model_save_freq
        self.frames = frames
        self.loss_version = "org"

        # set init values
        self.config = config if config is not None else default_config
        self.config.update(kwargs)
        self.dev_mode = False
        self.wandb_run = None
        self._class = "[Predictor]"

        # set model
        # print(f"Setting model to {model}")
        self.set_model("rPPGnet" if model is None else model)
        self.set_ex_model("FaceHRNet09V4ELU" if ex_model is None else ex_model)
        self.set_est_model(
            "SNREstimatorNetMonteCarlo" if est_model is None else est_model
        )

        # set wandb
        if use_wandb:
            self.set_logger(project=project)

        # set dataset
        self.set_dataset()

    def load_model(self, path, model_name="model.pth"):
        print(f"{self._class} Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = "logs/Eulerian_Magnification/models/latest_from_wandb"
            path = path[6:]
            print(f"{self._class} Downloading model from wandb: {path}")
            path = download_model(path, dl_path)

        self.model.load_state_dict(
            torch.load(path + "/" + model_name, map_location=torch.device(self.device))
        )

    def set_model(self, model):
        print(f"{self._class} Setting model to {model}")

        transfer_learning = model.lower() in ["resnet18"]
        # if model.lower() == "resnet18":
        #     self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        #     self.model.name = "resnet18"
        # else:
        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")

        if transfer_learning:
            print("hi")
            self.model = model(
                dropout=self.config["dropout"],
                batchnorm=self.config["batchnorm"],
                finetune=self.config["finetune"],
            )
        else:
            self.model = model()

        self.model.to(self.device)

    def set_ex_model(self, model):
        print(f"{self._class} Setting ex model to {model}")

        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")

        self.ex_model = model(rgb=True)
        # self.ex_model.setup([1,0,0,0,0,0,0,0,0,0,0,0], 3, 3, 300)
        self.ex_model.to(self.device)

    def set_est_model(self, model):
        print(f"{self._class} Setting est model to {model}")

        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")

        self.est_model = model()
        # self.est_model.setup(ex)
        self.est_model.setup([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 3, 3)
        self.est_model.to(self.device)

    def set_dataset(self):
        # self.data_train = BenchDataset(**self.config.get("train_dataset_kwargs", {}), purpose="train")
        # self.data_test = BenchDataset(**self.config.get("test_dataset_kwargs", {}), purpose="test")
        # self.data_val = BenchDataset(**self.config.get("test_dataset_kwargs", {}), purpose="val")
        self.data_train = CustomDataset(
            **self.config.get("train_dataset_kwargs", {}), purpose="train"
        )
        self.data_test = CustomDataset(
            **self.config.get("test_dataset_kwargs", {}), purpose="test"
        )
        self.data_val = CustomDataset(
            **self.config.get("test_dataset_kwargs", {}), purpose="val"
        )

        self.train_loader = self.data_train.get_dataloader(
            **self.config.get("train_dataloader_kwargs", {})
        )
        self.test_loader = self.data_test.get_dataloader(
            **self.config.get("train_dataloader_kwargs", {})
        )
        self.val_loader = self.data_val.get_dataloader(
            **self.config.get("train_dataloader_kwargs", {})
        )

        # self.train_loader, self.val_loader = self.data_train.get_dataloader(**self.config.get("train_dataloader_kwargs", {}))

    def set_optimizer(self):
        optimizer = self.config.get("optimizer")

        # set default lr if not set
        if optimizer.lower() == "sgd" and "lr" not in self.config["optimizer_kwargs"]:
            self.config["optimizer_kwargs"]["lr"] = 0.01

        # set optimizer
        self.optimizer = torch.optim.__dict__.get(optimizer)(
            self.model.parameters(), **self.config.get("optimizer_kwargs", {})
        )
        # self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(),)
        self.extractor_optimizer = torch.optim.Adam(
            self.ex_model.parameters(), lr=0.0001, weight_decay=0.000
        )
        self.estimator_optimizer = torch.optim.Adam(
            self.est_model.parameters(), lr=0.1, weight_decay=0.000
        )

        if self.config.get("use_scheduler", True):
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.1, step_size = int(5+0.2*self.config["num_epochs"]))
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=3,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
                verbose=False,
            )
        else:
            self.scheduler = None

    def save_model(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(
                self.name + "_model",
                type="model",
                description=f"model trained on hotdog dataset after {epoch} epochs",
            )
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)

    def set_logger(self, **kwargs):
        # overwrite defaults with parsed arguments
        wandb_settings = wandb_defaults.copy()
        wandb_settings.update(kwargs)
        wandb_settings.update(
            {"dir": "logs/" + wandb_settings.get("project") + "/" + self.name}
        )  # create log dir
        wandb_settings.update(
            {"name": self.name, "group": self.model.name}
        )  # set run name

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb
        self.wandb_run = wandb.init(**wandb_settings, config=self.config)

        # setup wandb config
        self.config = self.wandb_run.config

        # watch model
        self.wandb_run.watch(self.model)

    def prepare(self, cuda_device):
        # set seed
        if self.config.get("seed") is not None:
            torch.manual_seed(self.config.get("seed", 0))

        # set visible cuda devices
        if self.device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(cuda_device).strip("[]").replace(" ", "")
            )

        # set loss function
        criterion_pearson = Neg_Pearson()
        criterion_pearson_2 = Neg_Pearson2()
        criterion_binary = torch.nn.BCELoss()
        criterion_L1 = torch.nn.L1Loss()
        criterion_mse = torch.nn.MSELoss()
        self.loss_fun = lambda output, target: criterion_pearson_2(output, target)
        self.loss_fun_skin = lambda output, target: criterion_binary(output, target)
        self.loss_fun_L1 = lambda output, target: criterion_L1(output, target)
        self.loss_fun_mse = lambda output, target: criterion_mse(output, target)

        # set optimizer
        self.set_optimizer()

        # set best acc
        self.best_test_loss = 3

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def train_step(self, data, target, target_skin):
        # send data to device
        data, target, target_skin = (
            data.to(self.device),
            target.to(self.device),
            target_skin.to(self.device),
        )

        # Zero the gradients computed for each weight
        self.optimizer.zero_grad()
        # self.estimator_optimizer.zero_grad()
        # self.extractor_optimizer.zero_grad()

        # Forward pass your image through the network
        (
            skin_map,
            rPPG_aux,
            rPPG,
            rPPG_SA1,
            rPPG_SA2,
            rPPG_SA3,
            rPPG_SA4,
            x_visual6464,
            x_visual3232,
        ) = self.model(data)
        # rPPG2 = self.ex_model(torch.swapaxes(data.squeeze(0), 0, 1)).squeeze().unsqueeze(dim=0).unsqueeze(dim=0)
        # output = self.ex_model(data.squeeze(0)).squeeze().unsqueeze(dim=0).unsqueeze(dim=0)
        # print(output)
        # outputs = (output - torch.mean(output)) / torch.std(output)
        # fs = 120
        # lowcut = 0.1
        # highcut = 15

        # filtered_outputs = self.butter_bandpass_filter(outputs.detach().cpu().numpy(), lowcut, highcut, fs, order=4)
        # filtered_outputs = (filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)
        # print(torch.mean(torch.from_numpy(filtered_outputs).view(1, 300), dim=-1))

        # x - torch.mean(x, dim=-1).unsqueeze(dim=-1)
        # print(rPPG2.size())

        # output = self.est_model(output).squeeze()
        # print(output)
        # output = torch.mul(output.item(), 100)
        # print(output)
        # print("Estimated HR from ex/est model: ", output.item())

        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
        rPPG_SA1 = (rPPG_SA1 - torch.mean(rPPG_SA1)) / torch.std(rPPG_SA1)  # normalize2
        rPPG_SA2 = (rPPG_SA2 - torch.mean(rPPG_SA2)) / torch.std(rPPG_SA2)  # normalize2
        rPPG_SA3 = (rPPG_SA3 - torch.mean(rPPG_SA3)) / torch.std(rPPG_SA3)  # normalize2
        rPPG_SA4 = (rPPG_SA4 - torch.mean(rPPG_SA4)) / torch.std(rPPG_SA4)  # normalize2
        rPPG_aux = (rPPG_aux - torch.mean(rPPG_aux)) / torch.std(rPPG_aux)  # normalize2

        # rPPG = self.ex_model(rPPG.squeeze().unsqueeze(dim=0).unsqueeze(dim=0))
        # rPPG = rPPG.detach().cpu().squeeze(0).numpy()
        # target = torch.mean(target.float()[0]).cuda()
        # print("GT mean HR: ", target.item())

        # Compute the loss
        # assert len(rPPG.detach().cpu().numpy()[0]) == len(target.cpu().numpy()[0])

        loss_binary = self.loss_fun_skin(skin_map, target_skin)
        loss_ecg = self.loss_fun(rPPG.to(torch.float64).to(self.device), target)
        loss_ecg1 = self.loss_fun(rPPG_SA1.to(torch.float64).to(self.device), target)
        loss_ecg2 = self.loss_fun(rPPG_SA2.to(torch.float64).to(self.device), target)
        loss_ecg3 = self.loss_fun(rPPG_SA3.to(torch.float64).to(self.device), target)
        loss_ecg4 = self.loss_fun(rPPG_SA4.to(torch.float64).to(self.device), target)
        loss_ecg_aux = self.loss_fun(rPPG_aux.to(torch.float64).to(self.device), target)
        if self.loss_version == "org":
            loss = (
                0.1 * loss_binary
                + 0.5 * (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux)
                + loss_ecg
            )
        elif self.loss_version == "peaks":
            model_output = rPPG.detach().cpu().numpy()[0]
            target = target.detach().cpu().numpy()[0]
            model_peaks = find_peaks(model_output, height=(1, None), distance=10)[0]
            target_peaks = find_peaks(target, height=(1, None), distance=10)[0]
            model_t = np.zeros(self.frames)
            target_t = np.zeros(self.frames)
            model_t[model_peaks] = 1
            target_t[target_peaks] = 1
            model_hr = 25 / np.diff(model_peaks) * 60
            target_hr = 25 / np.diff(target_peaks) * 60
            if len(model_peaks) < 2:
                model_hr = [10000]
            print("Model hr: ", model_hr)
            print("Target hr: ", target_hr)
            loss = torch.tensor(
                abs(np.mean(model_hr) - np.mean(target_hr)), requires_grad=True
            )

        else:
            loss = self.loss_fun_mse(rPPG.to(torch.float64).to(self.device), target)

        # train_abs_err = (torch.abs(output - target)).view(1)
        # train_sq_err = ((output - target) ** 2).view(1)
        # loss = train_abs_err
        # print(loss)
        # Backward pass through the network
        loss.backward()

        # Update the weights
        self.optimizer.step()

        # return loss_binary, loss_ecg, loss
        # return train_abs_err, train_sq_err
        return (
            loss_binary,
            loss_ecg,
            loss_ecg1,
            loss_ecg2,
            loss_ecg3,
            loss_ecg4,
            loss_ecg_aux,
            loss,
        )

    def train(self, num_epochs=None, cuda_device=[0]):
        # prepare training
        num_epochs = self.config.get("num_epochs") if num_epochs is None else num_epochs
        self.prepare(cuda_device)
        print(f"{self._class} Starting training on {self.device.type}")

        for epoch in tqdm(range(num_epochs), unit="epoch"):
            train_loss = 0
            binary_loss = 0
            ecg_total_loss = 0
            ecg1_loss = 0
            ecg2_loss = 0
            ecg3_loss = 0
            ecg4_loss = 0
            ecg_aux_loss = 0
            self.epoch = epoch
            self.model.train()
            # self.est_model.train()
            # self.ex_model.train()
            count = 0
            pbar = tqdm(enumerate(self.train_loader), total=5)
            # pbar = tqdm(enumerate(self.train_loader), total=int(1969/85))
            for minibatch_no, (target_skin, data, target) in pbar:
                if target.shape[1] != self.frames:
                    # print(target.shape[1])
                    # print(self.frames)
                    break
                (
                    loss_binary,
                    loss_ecg,
                    loss_ecg1,
                    loss_ecg2,
                    loss_ecg3,
                    loss_ecg4,
                    loss_ecg_aux,
                    loss,
                ) = self.train_step(data, target, target_skin)
                # loss_binary, loss_ecg, loss = self.train_step(data, target, target_skin)
                # train_abs_err, train_sq_err = self.train_step(data, target, target_skin)
                count += 1
                train_loss += loss.item()
                binary_loss += loss_binary.item()
                ecg_total_loss += loss_ecg.item()
                ecg1_loss += loss_ecg1.item()
                ecg2_loss += loss_ecg2.item()
                ecg3_loss += loss_ecg3.item()
                ecg4_loss += loss_ecg4.item()
                ecg_aux_loss += loss_ecg_aux.item()

                print("Train loss: ", train_loss / (minibatch_no + 1))
                # break if dev mode
                if self.dev_mode:
                    break

            # Comput the train accuracy
            train_loss /= count
            binary_loss /= count
            ecg_total_loss /= count
            ecg1_loss /= count
            ecg2_loss /= count
            ecg3_loss /= count
            ecg4_loss /= count
            ecg_aux_loss /= count
            # train_loss /= len(self.train_loader)
            # binary_loss /= len(self.train_loader)
            # ecg_total_loss /= len(self.train_loader)
            # ecg1_loss /= len(self.train_loader)
            # ecg2_loss /= len(self.train_loader)
            # ecg3_loss /= len(self.train_loader)
            # ecg4_loss /= len(self.train_loader)
            # ecg_aux_loss /= len(self.train_loader)

            if self.verbose:
                print(f"{self._class} Train Loss: {train_loss:.3f}")

            # test
            (
                test_loss,
                test_loss_binary,
                test_loss_ecg,
                test_loss_ecg1,
                test_loss_ecg2,
                test_loss_ecg3,
                test_loss_ecg4,
                test_loss_ecg_aux,
            ) = self.test(validation=True)
            # test_loss, test_loss_binary, test_loss_ecg = self.test(validation=True)

            # take step in scheduler
            if self.scheduler:
                self.scheduler.step(test_loss)
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"Learning rate": self.scheduler.get_last_lr()[0]}, commit=False
                    )

            # Save model
            if self.model_save_freq is None:
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                    self.save_model(
                        f"logs/Eulerian/models/{self.name}/model.pth", epoch
                    )
            elif epoch % self.model_save_freq == 0:
                self.save_model(f"logs/Eulerian/models/{self.name}/model.pth", epoch)

            # log to wandb
            if self.wandb_run is not None:
                # self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log(
                    {
                        "Train metrics/Binary_Loss": binary_loss,
                        "Train metrics/ecg_Loss": ecg_total_loss,
                        "Train metrics/ecg1_Loss": ecg1_loss,
                        "Train metrics/ecg2_Loss": ecg2_loss,
                        "Train metrics/ecg3_Loss": ecg3_loss,
                        "Train metrics/ecg4_Loss": ecg4_loss,
                        "Train metrics/ecg_aux_Loss": ecg_aux_loss,
                        "Train metrics/Total_Loss": train_loss,
                        "epoch": epoch,
                        "Test metrics/Binary_Loss": test_loss_binary,
                        "Test metrics/ecg_Loss": test_loss_ecg,
                        "Test metrics/ecg1_Loss": test_loss_ecg1,
                        "Test metrics/ecg2_Loss": test_loss_ecg2,
                        "Test metrics/ecg3_Loss": test_loss_ecg3,
                        "Test metrics/ecg4_Loss": test_loss_ecg4,
                        "Test metrics/ecg_aux_Loss": test_loss_ecg_aux,
                        "Test metrics/Total_loss": test_loss,
                    }
                )

            # clear cache
            self.clear_cache()

        # log best val acc
        if self.wandb_run is not None:
            self.wandb_run.log(
                {"Best validation accuracy": self.best_test_loss}, commit=True
            )

    def test(self, validation=False):
        if validation:
            data_loader = self.val_loader
        else:
            self.prepare([0])
            data_loader = self.test_loader
        data_len = len(data_loader.dataset)
        print(data_loader)

        # Init counters
        test_loss = 0
        test_loss_binary = 0
        test_loss_ecg = 0
        test_loss_ecg1 = 0
        test_loss_ecg2 = 0
        test_loss_ecg3 = 0
        test_loss_ecg4 = 0
        test_loss_ecg_aux = 0

        # Test the model
        self.model.eval()
        # self.ex_model.eval()
        # self.est_model.eval()
        print("Validating...")
        count = 0
        inner_count = 0
        for target_skin, data, target in data_loader:
            inner_count += 1
            data = data.to(self.device)
            target_skin = target_skin.to(self.device)
            target = target.to(self.device)
            if target.shape[1] != self.frames:
                break
            count += 1
            with torch.no_grad():
                (
                    skin_map,
                    rPPG_aux,
                    rPPG,
                    rPPG_SA1,
                    rPPG_SA2,
                    rPPG_SA3,
                    rPPG_SA4,
                    x_visual6464,
                    x_visual3232,
                ) = self.model(data)

            # Update counters
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
            rPPG_SA1 = (rPPG_SA1 - torch.mean(rPPG_SA1)) / torch.std(
                rPPG_SA1
            )  # normalize2
            rPPG_SA2 = (rPPG_SA2 - torch.mean(rPPG_SA2)) / torch.std(
                rPPG_SA2
            )  # normalize2
            rPPG_SA3 = (rPPG_SA3 - torch.mean(rPPG_SA3)) / torch.std(
                rPPG_SA3
            )  # normalize2
            rPPG_SA4 = (rPPG_SA4 - torch.mean(rPPG_SA4)) / torch.std(
                rPPG_SA4
            )  # normalize2
            rPPG_aux = (rPPG_aux - torch.mean(rPPG_aux)) / torch.std(
                rPPG_aux
            )  # normalize2

            tlb = self.loss_fun_skin(skin_map, target_skin)
            tle = self.loss_fun(rPPG.to(torch.float64), target)
            tle1 = self.loss_fun(rPPG_SA1.to(torch.float64), target)
            tle2 = self.loss_fun(rPPG_SA2.to(torch.float64), target)
            tle3 = self.loss_fun(rPPG_SA3.to(torch.float64), target)
            tle4 = self.loss_fun(rPPG_SA4.to(torch.float64), target)
            tlea = self.loss_fun(rPPG_aux.to(torch.float64), target)

            loss = 0.1 * tlb + 0.5 * (tle1 + tle2 + tle3 + tle4 + tlea) + tle

            test_loss_binary += tlb
            test_loss_ecg += tle
            test_loss_ecg1 += tle1
            test_loss_ecg2 += tle2
            test_loss_ecg3 += tle3
            test_loss_ecg4 += tle4
            test_loss_ecg_aux += tlea

            if self.loss_version == "org":
                test_loss += loss
            else:
                test_loss += self.loss_fun_mse(rPPG.to(self.device), target)
            print("Test Loss: ", test_loss.item() / count)

            if inner_count == 3:
                print("saving a plot...")
                plt.figure(figsize=(10, 8))
                model_output = rPPG.cpu().numpy()[0]
                target = target.cpu().numpy()[0]
                plt.plot(model_output, label="model_output")
                plt.plot(target, label="target")
                # temp = pd.DataFrame([target, model_output], columns=["target", "model_output"])
                # temp.to_csv('temp.csv')
                # peaks = QRSDetectorOffline(ecg_data_path="temp.csv", verbose=False, log_data=False, plot_data=False, show_plot=False).qrs_peaks_indices
                peaks = find_peaks(target, height=(1, None), distance=10)[0]
                t_peaks = [target[i] for i in peaks]
                print("GT heartrates:", 25 / np.diff(peaks) * 60)
                plt.scatter(peaks, t_peaks, color="r", marker="o", label="Peaks")

                peaks = find_peaks(model_output, height=(1, None), distance=10)[0]
                t_peaks = [model_output[i] for i in peaks]
                print("Predicted heartrates:", 25 / np.diff(peaks) * 60)
                plt.scatter(peaks, t_peaks, color="r", marker="o", label="Peaks")

                plt.title(
                    "Test iteration {} for epoch {}. Loss: {}".format(
                        inner_count, self.epoch, loss
                    )
                )
                plt.xlabel("Frame nr")
                plt.legend()
                path = "model_figs/{}".format(self.name)
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(
                    "{}/signals_{}_{}.png".format(path, self.epoch, inner_count)
                )
                # if self.wandb_run is not None:
                #     self.wandb_run.log({"plots/{}".format(self.epoch):'{}/signals_{}_{}.png'.format(path, self.epoch, inner_count)
                # })

        # compute stats
        # test_loss /= len(data_loader)
        # test_loss_binary /= len(data_loader)
        # test_loss_ecg /= len(data_loader)
        # test_loss_ecg1 /= len(data_loader)
        # test_loss_ecg2 /= len(data_loader)
        # test_loss_ecg3 /= len(data_loader)
        # test_loss_ecg4 /= len(data_loader)
        # test_loss_ecg_aux /= len(data_loader)
        test_loss /= count
        test_loss_binary /= count
        test_loss_ecg /= count
        test_loss_ecg1 /= count
        test_loss_ecg2 /= count
        test_loss_ecg3 /= count
        test_loss_ecg4 /= count
        test_loss_ecg_aux /= count

        if self.verbose:
            print("Loss test: {test:.3f}".format(test=test_loss))

        return (
            test_loss,
            test_loss_binary,
            test_loss_ecg,
            test_loss_ecg1,
            test_loss_ecg2,
            test_loss_ecg3,
            test_loss_ecg4,
            test_loss_ecg_aux,
        )
        # return test_loss, test_loss_binary, test_loss_ecg

    # def sweep(self, **kwargs):
    #    sweep_configuration = sweep_defaults.copy()
    #   sweep_configuration.update(kwargs)

    #  sweep_id = wandb.sweep(
    #     sweep=sweep_configuration,
    #    project='Hotdog-sweeps'
    # )

    # Start sweep job.
    # wandb.agent(sweep_id, function=self.train, count=4)
    # os.system(f"wandb agent {sweep_id}")

    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")


if __name__ == "__main__":
    predictor = Predictor(
        project="Eulerian_mag",
        model="rPPGNet",
        frames=64,
        use_wandb=False,
        optimizer="Adam",
    )
    # classifier.dev_mode = True
    predictor.train(num_epochs=50)
    # classifier.sweep()

    # classifier.load_model("wandb:deepcomputer/Hotdog/Resnet18_finetune_model:v8", model_name="Resnet18_finetune.pth")
    # classifier.test(save_images=10)
