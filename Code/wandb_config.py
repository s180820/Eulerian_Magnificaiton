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