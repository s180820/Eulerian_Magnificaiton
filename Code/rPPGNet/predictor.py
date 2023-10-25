import torch
from torch.nn import functional as F
import torchvision
from tqdm import tqdm
import os
from _config import wandb_defaults, default_config
from dataloader import CustomDataset
import wandb
import numpy as np
from utils import download_model

from models import models
from loss_functions import Neg_Pearson

from datetime import datetime as dt

class Predictor:
    def __init__(self, project = "Eulerian_Magnification", name = None, model = None, config = None, use_wandb = True, verbose = True, show_test_images = False, model_save_freq = None, **kwargs):
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
        self.name = name if name is not None else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_test_images = show_test_images
        self.model_save_freq = model_save_freq

        # set init values
        self.config = config if config is not None else default_config
        self.config.update(kwargs)
        self.dev_mode = False
        self.wandb_run = None
        self.test_images = []

        # set model
        # print(f"Setting model to {model}")
        self.set_model("rPPGnet" if model is None else model)


        # set wandb
        if use_wandb:
            self.set_logger(project = project)


        # set dataset
        self.set_dataset()
        
    
    def load_model(self, path, model_name = "model.pth"):
        print(f"Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = "logs/Eulerian_Magnification/models/latest_from_wandb"
            path = path[6:]
            print(f"Downloading model from wandb: {path}")
            path = download_model(path, dl_path)
        
        self.model.load_state_dict(torch.load(path+"/"+model_name, map_location=torch.device(self.device)))           
        
    def set_model(self, model):
        print(f"Setting model to {model}")
        
        transfer_learning = model.lower() in ["resnet18"]
        # if model.lower() == "resnet18":
        #     self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        #     self.model.name = "resnet18"
        # else:
        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")
        
        if transfer_learning:
            self.model = model(dropout = self.config["dropout"], batchnorm = self.config["batchnorm"], finetune = self.config["finetune"])
        else:
            self.model = model()
        
        self.model.to(self.device)


    def set_dataset(self):
        self.data_train = CustomDataset(**self.config.get("train_dataset_kwargs", {}))
        self.data_test = CustomDataset(**self.config.get("test_dataset_kwargs", {}))
             
        self.train_loader = self.data_train.get_dataloader(**self.config.get("train_dataloader_kwargs", {}))
        self.test_loader = self.data_test.get_dataloader(**self.config.get("test_dataloader_kwargs", {}))

    def set_optimizer(self):
        optimizer = self.config.get("optimizer")
        
        # set default lr if not set
        if optimizer.lower() == "sgd" and "lr" not in self.config["optimizer_kwargs"]:
            self.config["optimizer_kwargs"]["lr"] = 0.01
            
        # set optimizer
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), **self.config.get("optimizer_kwargs", {}))
        # self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), )
        
        
        
        #if self.config.get("use_scheduler", True):
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.1, step_size = int(5+0.2*self.config["num_epochs"]))
         #   self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        #else:
         #   self.scheduler = None

    def save_model(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(self.name + "_model", type="model", description=f"model trained on hotdog dataset after {epoch} epochs")
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)

    def set_logger(self, **kwargs):
        # overwrite defaults with parsed arguments
        wandb_settings = wandb_defaults.copy()
        wandb_settings.update(kwargs)
        wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + "/" + self.name}) # create log dir
        wandb_settings.update({"name" : self.name, "group": self.model.name}) # set run name

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb
        self.wandb_run = wandb.init(**wandb_settings, config = self.config)

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
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip("[]").replace(" ", "")

        # set loss function
        criterion_pearson = Neg_Pearson()
        self.loss_fun = lambda output, target: criterion_pearson(output, target)
        self.loss_fun_skin = lambda output, target: F.mse_loss(output, target)

        # set optimizer
        self.set_optimizer()
        
        # set best acc
        self.best_val_acc = 0


    def train_step(self, data, target, target_skin):
        # send data to device
        data, target, target_skin = data.to(self.device), target.to(self.device), target_skin.to(self.device)

        #Zero the gradients computed for each weight
        self.optimizer.zero_grad()

        #Forward pass your image through the network
        skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232  = self.model(data)
        rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
        rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) /torch.std(rPPG_SA1)	 	# normalize2
        rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) /torch.std(rPPG_SA2)	 	# normalize2
        rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) /torch.std(rPPG_SA3)	 	# normalize2
        rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) /torch.std(rPPG_SA4)	 	# normalize2
        rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) /torch.std(rPPG_aux)	 	# normalize2

        #Compute the loss
        loss_binary = self.loss_fun_skin(skin_map, target_skin) 
        loss_ecg = self.loss_fun(rPPG, target)
        loss_ecg1 = self.loss_fun(rPPG_SA1, target)
        loss_ecg2 = self.loss_fun(rPPG_SA2, target)
        loss_ecg3 = self.loss_fun(rPPG_SA3, target)
        loss_ecg4 = self.loss_fun(rPPG_SA4, target)
        loss_ecg_aux = self.loss_fun(rPPG_aux, target)

        loss = 0.1*loss_binary +  0.5*(loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
        #Backward pass through the network
        loss.backward()

        #Update the weights
        self.optimizer.step()


        return loss_binary, loss_ecg, loss_ecg1, loss_ecg2, loss_ecg3, loss_ecg4, loss_ecg_aux, loss


    def train(self, num_epochs=None, cuda_device = [0]):
        # prepare training
        num_epochs = self.config.get("num_epochs") if num_epochs is None else num_epochs
        self.prepare(cuda_device)
        print(f"Starting training on {self.device.type}")

        for epoch in tqdm(range(num_epochs), unit='epoch'):
            
            train_loss = 0
            binary_loss = 0
            ecg_total_loss = 0
            ecg1_loss = 0
            ecg2_loss = 0
            ecg3_loss = 0
            ecg4_loss = 0
            ecg_aux_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for minibatch_no, (target_skin, data, target) in pbar:
                loss_binary, loss_ecg, loss_ecg1, loss_ecg2, loss_ecg3, loss_ecg4, loss_ecg_aux, loss = self.train_step(data, target, target_skin)
                
                train_loss += loss.item()
                binary_loss += loss_binary
                ecg_total_loss += loss_ecg
                ecg1_loss += loss_ecg1
                ecg2_loss += loss_ecg2
                ecg3_loss += loss_ecg3
                ecg4_loss += loss_ecg4
                ecg_aux_loss += loss_ecg_aux

                # break if dev mode
                if self.dev_mode:
                    break

            #Comput the train accuracy
            train_loss /= len(self.train_loader)
            binary_loss /= len(self.train_loader)
            ecg_total_loss /= len(self.train_loader)
            ecg1_loss /= len(self.train_loader)
            ecg2_loss /= len(self.train_loader)
            ecg3_loss /= len(self.train_loader)
            ecg4_loss /= len(self.train_loader)
            ecg_aux_loss /= len(self.train_loader)
            
            if self.verbose:
                print("Train Loss: {train:.1f}%".format(train=train_loss))
            
            # test 
            #val_acc, val_loss, conf_mat = self.test(validation=True)
            
            # take step in scheduler
            #if self.scheduler:
             #   self.scheduler.step(val_loss)
              #  if self.wandb_run is not None:
               #     self.wandb_run.log({"Learning rate" : self.scheduler.get_last_lr()[0]}, commit = False)
                
            
            # Save model
            #if self.model_save_freq is None:
             #   if val_acc > self.best_val_acc:
              #      self.best_val_acc = val_acc
               #     self.save_model(f"logs/Eulerian/models/{self.name}/model.pth", epoch)
            #elif epoch % self.model_save_freq == 0:
             #   self.save_model(f"logs/Eulerian/models/{self.name}/model.pth", epoch)
            
            
            # log to wandb
            if self.wandb_run is not None:
                #self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log({
                    "Train metrics/Binary_Loss":  binary_loss,
                    "Train metrics/ecg_Loss":  ecg_total_loss,
                    "Train metrics/ecg1_Loss":  ecg1_loss,
                    "Train metrics/ecg2_Loss":  ecg2_loss,
                    "Train metrics/ecg3_Loss":  ecg3_loss,
                    "Train metrics/ecg4_Loss":  ecg4_loss,
                    "Train metrics/ecg_aux_Loss":  ecg_aux_loss,
                    "Train metrics/Total_Loss":  loss,
                    "epoch":        epoch,
                })
            
            # clear cache
            self.clear_cache()
            
        # log best val acc
        if self.wandb_run is not None:
            self.wandb_run.log({"Best validation accuracy": self.best_val_acc}, commit = True)
            
            
    def test(self, validation = False, save_images = 0):  
        if validation:
            data_loader = self.val_loader
        else:
            self.prepare([0])
            data_loader = self.test_loader
        data_len = len(data_loader.dataset)
        
        print("Performing test with {} images".format(data_len))
        
        # Init counters
        test_correct = 0
        test_loss = 0
        
        # conf matrix
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        
        # Test the model
        self.model.eval()
        for data, target in data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            predicted = output.argmax(1).cpu()

            # Update counters
            test_loss += self.loss_fun(output.cpu(), target).item()
            
            # save images
            if save_images > 0:
                self.save_images(data, target, predicted, output, save_images)

        # compute stats
        test_loss /= len(data_loader)

        # calculate confusion matrix      
        test_loss /= len(data_loader)
        # conf_mat = {"true_positive": true_positive/data_len, "true_negative": true_negative/data_len, "false_positive": false_positive/data_len, "false_negative": false_negative/data_len}
        conf_mat = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative}
        
        if self.verbose:
            print("Accuracy test: {test:.1f}%".format(test=test_acc))

        if self.show_test_images:
            self.create_test_images(data, target, predicted, output)

        return test_acc, test_loss, conf_mat


   # def sweep(self, **kwargs):
    #    sweep_configuration = sweep_defaults.copy()
     #   sweep_configuration.update(kwargs)

      #  sweep_id = wandb.sweep(
       #     sweep=sweep_configuration,
        #    project='Hotdog-sweeps'
        #)

        # Start sweep job.
        # wandb.agent(sweep_id, function=self.train, count=4)
        #os.system(f"wandb agent {sweep_id}")

    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")


if __name__ == "__main__":
    predictor = Predictor(project="Eulerian_mag", name = "rPPGNet", 
                                  show_test_images=False, model = "rPPGNet", use_wandb=True, optimizer = "Adam",)
    # classifier.dev_mode = True
    predictor.train(num_epochs=50)
    # classifier.sweep()


    # classifier.load_model("wandb:deepcomputer/Hotdog/Resnet18_finetune_model:v8", model_name="Resnet18_finetune.pth")
    # classifier.test(save_images=10)
    
    
    