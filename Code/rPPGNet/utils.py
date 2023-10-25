import wandb
import torch.optim as optim

def download_model(full_name:str, download_path:str = None):
    """
    Downloads the agent from wandb.
    
    Parameters
    ----------
    full_name: str
        The full name of the wandb artifact to download.
    download_path: str
        The path to download the artifact to. The agent will be downloaded to `download_path/model.zip`.
    
    Returns
    -------
    model_path: str
        The path to the downloaded agent. This is `download_path/model`.
    """
    api = wandb.Api() # start wandb api
    artifact = api.artifact(full_name) # load artifact
    path = artifact.download(download_path) # download artifact
 
    return path 

class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return self._last_lr