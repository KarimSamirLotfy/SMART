
* Running bash install_pyg.sh
ERROR: torch_cluster-1.6.0+pt112cu113-cp310-cp39-linux_x86_64.whl is not a supported wheel on this platform.
ERROR: torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl is not a supported wheel on this platform.
ERROR: torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl is not a supported wheel on this platform.
ERROR: torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl is not a supported wheel on this platform.


# Arch
* Tokenization
    Agents have predefeined tokens. each veichle type has 2048 tokens. each token is 4 timestps of x, y so shape is 2048, 4, 2

* Averything is using Torch geometric. so all batching is creating a huge map where different scences and agents are disconnected. then they use other techqiues in torch_geometric, to be able to insuer query centric comminication. 

* It is simly, create edges ebtweetn things that should interact, get distance between them. use query centric attention on thngs that interact only. output. 

* This model is multi-task. a pseudo task for training is Road next token predictions on the polylines or points. 

* this makes it harder to visualise
# TODO
[] Create cisualsations
[] Create Metrics


# How to actually turn this into a diffusion model
1. map encoding stays the same. it is it's own module that encdes the information
2. the agent encoding can be split into 3 steps
    a. embedding agent into feat_a and returns agent_token_traj. which is the translation of the token to a 4 step movmenet
    b. then we build some temporal edges. just math explaining what gets to attend what
    c. And we also handlel batches. just math
    d. build interaction edges betweeen the agents and each other
    e. build interaction edges betwen aensts and other agents
    ### HERE WE CAN START LATENT DIFFUSION
    # input to preturb will be the feat_a as it is the latent.
    f. then the entire feat_a which is num_agents, timesteps, hidden_dim is flattened. to (-1, hidden_dim). so it becomes a huge number of hidden dims. 
    
    ### this is where all the attention happenes. by choosiing whihc edges talk to wihich it is easy to just attend.

    g. the feat_a is now of size num_agents, 18, 128
    ### HERE WE GET THE LOSS.
    ## extract the 
    h. then token_predict_head turns each 128 to 2048 which is then the softmax of the action token to choose. nothing to see here


# TRaining
8 Once can barely do batch size of 8. on wonder we need A100s

### ERRORS
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]

  | Name          | Type             | Params
---------------------------------------------------
0 | encoder       | SMARTDecoder     | 7.2 M 
1 | minADE        | minADE           | 0     
2 | minFDE        | minFDE           | 0     
3 | TokenCls      | TokenCls         | 0     
4 | waymo_metrics | WaymoMetrics     | 0     
5 | cls_loss      | CrossEntropyLoss | 0     
6 | map_cls_loss  | CrossEntropyLoss | 0     
---------------------------------------------------
7.2 M     Trainable params
0         Non-trainable params
7.2 M     Total params
28.605    Total estimated model params size (MB)
/home/k.lotfy/anaconda3/envs/smart/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:261: UserWarning: You requested to overfit but enabled train dataloader shuffling. We are turning off the train dataloader shuffling for you.
  rank_zero_warn(
/home/k.lotfy/anaconda3/envs/smart/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:432: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 64 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
/home/k.lotfy/anaconda3/envs/smart/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:280: PossibleUserWarning: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
  rank_zero_warn(
/home/k.lotfy/anaconda3/envs/smart/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:432: PossibleUserWarning: The dataloader, val_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 64 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Epoch 0: 100%|█| 1/1 [01:01<002024-11-04 00:39:28.110780: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED: initialization error
2024-11-04 00:39:28.110846: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: dnde-crd-ltl-mlops
2024-11-04 00:39:28.110853: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: dnde-crd-ltl-mlops
2024-11-04 00:39:28.111357: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: NOT_FOUND: was unable to find libcuda.so DSO loaded into this program
2024-11-04 00:39:28.111403: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 535.183.6
2024-11-04 00:39:31.188187: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...

# torch cuda 11.3 website for tenfolow says u need 12.3
* for waymo we have waymo-open-dataset-tf-2-12-0==1.6.4 newest version or pip install waymo-open-dataset-tf-2-11-0==1.6.1 as the latest release needs python==3.10
* TF req: cuda toolkit 12.3
* cuDNN fir cuda-toolkit-12 python3 -m pip install nvidia-cudnn-cu12 https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html
* pip install cuda-python==12.3.0
* pip install tensorflow[and-cuda]

* torch only have 12.1 and 12.4 will try with 12.4 maybe it is backward compatible pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
* python -m pip install lightning
* for waymo we have waymo-open-dataset-tf-2-12-0==1.6.4 newest version or pip install waymo-open-dataset-tf-2-11-0==1.6.1 as the latest release needs python==3.10
* get the correct torch eometric packages

## Finally found porblem. 
* As I preprocesssed the data, i added the senario bytes to the tenosr. this works for cpu but not gpu.
* decision, validation set works as normal, but i will create another thread that loads the model to cpu and shows the results
* FOR REFRENCE a batch size of 3 and allowing for vis and metrics. it takes 2:00 min to finish. AKA THIS IS NOT SUSTAINIABLE. 

# PREFORMANCE
batch_size of 4, mini-waymo, epoch = 20 with validaiton
batch=8, mini-waymo, epoch= 20 with validaiton
* Also there is a memroy leack. as the GPU ram is getting more and more full. hence it may be impossible to have huge batch sizes.