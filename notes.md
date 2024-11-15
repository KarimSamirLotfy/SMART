# Arch
* Tokenization
    Agents have predefeined tokens. each veichle type has 2048 tokens. each token is 4 timestps of x, y so shape is 2048, 4, 2

* Averything is using Torch geometric. so all batching is creating a huge map where different scences and agents are disconnected. then they use other techqiues in torch_geometric, to be able to insuer query centric comminication. 

* It is simly, create edges ebtweetn things that should interact, get distance between them. use query centric attention on thngs that interact only. output. 

* This model is multi-task. a pseudo task for training is Road next token predictions on the polylines or points. 

* Tokenization uses k-disks, which is a technqiue to do get the tokens from the dataset. 



# How to actually turn this into a diffusion model
1. Discreet diffusion
    * Do diffusion using categorical functions. this can be done using discreet diffusion pipeline. 
2. do diffusion on contrinous actiion space of accelerationa nd yaw. this could be good as the initla env is cts. but you loose on the tokenizer which is huge. 

# TRaining
* 8 Once can barely do batch size of 8. on wonder we need A100s
* Training the OG model needed 23 hours on 32 V100s GPU
* our training run. 1 epoch takes around 50 hours. 



# PREFORMANCE
batch_size of 4, mini-waymo, epoch = 20 with validaiton
batch=8, mini-waymo, epoch= 20 with validaiton
* Also there is a memroy leack. as the GPU ram is getting more and more full. hence it may be impossible to have huge batch sizes.

# TODO
[x] write diffusion discreet code
[x] get the path for visualsations and metrics
[] make sure the forward function knows it is doing inference. 

## Hyperparameters to think about in order
1. noise scheduler [linear, cosine constant, exponential]
    * constant has the best loss drop. and the worse generaitons. it has bad vis and bad metrics
    * cosine and exponential seem to be equal wth linear doing tradeoffs. expo is a tad better vis wise. but we will use cosine. as it is the state of the art espacially in the paper. VBD. but let us remember that expo was a bit better vis wise. 
2. num diffusion steps
3. alphas and betas
4. MCMC corrector. 