wandb_defaults = {
	# "sync_tensorboard":True, 
	"reinit":True,
	"entity" : "digitalmedia",
	# "name" : self.run_name,
	"project" : "Eulerian_magnification", # wandb project name, each project correpsonds to an experiment
	# "dir" : "logs/" + "GetStarted", # dir to store the run in
	# "group" : self.agent_name, # uses the name of the agent class
	"save_code" : True,
	"mode" : "online",
}

default_config = {
    # "model"	: "SimpleCNN",
    "seed":0,
	"optimizer" : "Adam",
	"loss_fun" : "rPPGnet",
	"num_epochs" : 25,
    "use_scheduler" :True,
	### model params
	"dropout": 0.5,
	"batchnorm": False,
	"finetune": False,
	# kwargs
	"optimizer_kwargs" : {"lr": 0.0001},
	"train_dataset_kwargs" : {#"json_file" :'Data/json_structure',
                              "root_dir": "/work3/s174159/Bench_data/",
                              "frames": 64,
                              "verbosity": True
							  },
	"test_dataset_kwargs" : {#"json_file" :'Data/json_structure',
                              "root_dir": "/work3/s174159/Bench_data/",
                              "frames": 64},
	"train_dataloader_kwargs" : {},
	"test_dataloader_kwargs" : {},
}

#sweep_defaults = {
 #   'program': "hotdog/cli.py",
  #  'method': 'bayes',
    # 'name': 'sweep',
   # 'metric': {
    #    'goal': 'maximize', 
     #   'name': 'Validation metrics/val_acc'
     #   },
    #'parameters': {
     #   'optimizer': {"values" : [ 'Adam', 'SGD' ]},
      #  'lr': {'min': 0.0001, 'max': 0.1},
       # 'data_augmentation': {'values': [True, False]},
        #'batchnorm': {'values': [True, False]},
        #'dropout': {'values': [0.0, 0.25, 0.5]},
        # 'batch_size': {'values': [16, 32, 64]},
        # 'num_epochs': {'max': 20, "min": 1, "distribution" : "int_uniform"},
        # 'lr': {'max': 0.1, 'min': 0.0001}
     #},
    # "early_terminate": {
	# 	"type": "hyperband",
	# 	"min_iter": 1,
	# 	"max_iter": 3,
	# }
#}