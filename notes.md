# Arch
* Tokenization
    Agents have predefeined tokens. each veichle type has 2048 tokens. each token is 4 timestps of x, y so shape is 2048, 4, 2

* Averything is using Torch geometric. so all batching is creating a huge map where different scences and agents are disconnected. then they use other techqiues in torch_geometric, to be able to insuer query centric comminication. 

* It is simly, create edges ebtweetn things that should interact, get distance between them. use query centric attention on thngs that interact only. output. 

* This model is multi-task. a pseudo task for training is Road next token predictions on the polylines or points. 

* Tokenization uses k-disks, which is a technqiue to do get the tokens from the dataset. 
# TODO
[x] Create cisualsations
[x] Create Metrics


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

### Issues with diffusion models in this pace
1. if we simply use diffsuion model right after the agent_token embedding. what is he ground truth. ...
* first idea. noise the feat_a which is just mlp that turns a lot of info to 128. then move it into the denoiser. then tell it to return the noise, and now you have the denoised_noise and noise. 
* Issue, unlike normal latent diffusion, we dont have encoder decoder arch
agent_info --mlp--> agent_embedding -----Attention, map, agents, time----> new_agent_embedding --mlp---> tokens

2. Use a difusion technqes that actually works on sequence discreet data. 2 options discreet diffusion. and embedding based diffusion. only mebdding allows for classfier conditionng which is baseically sampling. 
* 

so we are not encoding, decoding
in stable dffisuon the latents, the encoder gives latent. then we do on latent then we return the latent to decoder. but notice there is a ltent. the latent that goes in is the correct answer. here no, the latent that goes in is not the ccrect answer. so telling the model denoise the noise i added to the agent_embeddings does not make sense. as the model will do it's best to recover the agent embeddings instead of creaing the new_agent_embeddings that actually have what we want and need. 




# TRaining
* 8 Once can barely do batch size of 8. on wonder we need A100s
* Training the OG model needed 23 hours on 32 V100s GPU
* our training run. 1 epoch takes around 50 hours. 

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