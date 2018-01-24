Usage:
(optional) 1. change model structure in the last of lm.py, which has similar usage as popular ML frameworks.
2. run:
  python lm.py [-b batch_size] [-e num_epoch] [-m momentum] [-l learning_rate] [-d weight_decay] 
  
The default "python lm.py" will run a Tanh 4-gram model with 128 hidden units and 16 embedding dimension.
