mkdir pyg_depend && cd pyg_depend
wget https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_cluster-1.6.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.2+pt24cu124-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.18+pt24cu124-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.2+pt24cu124-cp310-cp310-linux_x86_64.whl
python3 -m pip install torch_cluster-1.6.0+pt112cu113-cp310-cp310-linux_x86_64.whl
python3 -m pip install torch_scatter-2.1.0+pt112cu113-cp310-cp310-linux_x86_64.whl
python3 -m pip install torch_sparse-0.6.16+pt112cu113-cp310-cp310-linux_x86_64.whl
python3 -m pip install torch_spline_conv-1.2.1+pt112cu113-cp310-cp310-linux_x86_64.whl
python3 -m pip install torch_geometric
