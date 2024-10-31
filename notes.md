
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

* this makes it harder to visualise
# TODO
[] Create cisualsations
[] Create Metrics

