For Citeseer-Full and Cora-ML datasets, PyG loads them as undirected by default. To utilize these datasets in their directed form, a slight modification is required in the PyG local installation. Please comment out the line in the file located at:edge_index = to_undirected(edge_index, num_nodes=x.size(0))

/miniconda
3/envs/your_env/lib/python3.10/site-packages/torch_geometric/io/npz.py
