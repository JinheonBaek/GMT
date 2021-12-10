# Accurate Learning of Graph Representations with Graph Multiset Pooling
Official Code Repository for the paper "Accurate Learning of Graph Representations with Graph Multiset Pooling" (ICLR 2021) : https://arxiv.org/abs/2102.11533.

In this repository, we implement both *Graph Multiset Pooling* (GMPool) and *Graph Multiset Transformer* (GMT), proposed in our paper.

* Note that our GMT is also featured in PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GraphMultisetTransformer, wherein you can more easily implement each block of our model. See the example [here](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_gmt.py).

## Abstract

Graph neural networks have been widely used on modeling graph data, achieving impressive results on node classification and link prediction tasks. Yet, obtaining an accurate representation for a graph further requires a pooling function that maps a set of node representations into a compact form. A simple sum or average over all node representations considers all node features equally without consideration of their task relevance, and any structural dependencies among them. Recently proposed hierarchical graph pooling methods, on the other hand, may yield the same representation for two different graphs that are distinguished by the Weisfeiler-Lehman test, as they suboptimally preserve information from the node features. To tackle these limitations of existing graph pooling methods, we first formulate the graph pooling problem as a multiset encoding problem with auxiliary information about the graph structure, and propose a Graph Multiset Transformer (GMT) which is a multi-head attention based global pooling layer that captures the interaction between nodes according to their structural dependencies. We show that GMT satisfies both injectiveness and permutation invariance, such that it is at most as powerful as the Weisfeiler-Lehman graph isomorphism test. Moreover, our methods can be easily extended to the previous node clustering approaches for hierarchical graph pooling. Our experimental results show that GMT significantly outperforms state-of-the-art graph pooling methods on graph classification benchmarks with high memory and time efficiency, and obtains even larger performance gain on graph reconstruction and generation tasks.

### Contribution of this work

* We treat a graph pooling problem as a multiset encoding problem, under which we consider relationships among nodes in a set with several attention units, to make a compact representation of an entire graph only with one global function, without additional message-passing operations.
* We show that existing GNN with our parametric pooling operation can be as powerful as the WL test, and also be easily extended to the node clustering approaches with learnable clusters.
* We extensively validate GMT for graph classification, reconstruction, and generation tasks on synthetic and real-world graphs, on which it largely outperforms most graph pooling baselines.

## Dependencies

* Python 3.7
* PyTorch 1.4
* PyTorch Geometric 1.4.3

## Run

To run the proposed model in the paper, run following commands:

* Graph Classification on TU datasets, including D&D, PROTEINS, MUTAG, IMDB-BINARY, IMDB-MULTI, and COLLAB (See the [script file](https://github.com/JinheonBaek/GMT/blob/main/scripts/classification_TU.sh) for the detailed experimental setup on each dataset).
* First, and second arguments denote the gpu_id and experiment_number.

```sh
sh ./scripts/classification_TU.sh 0 000
```

* Graph Classification on OGB datasets, including HIV, Tox21, ToxCast, and BBBP (See the [script file](https://github.com/JinheonBaek/GMT/blob/main/scripts/classification_OGB.sh) for the detailed experimental setup on each dataset).
* First, and second arguments denote the gpu_id and experiment_number.

```sh
sh ./scripts/classification_OGB.sh 0 000
```

* Graph Reconstruction on the ZINC dataset (See the [script file](https://github.com/JinheonBaek/GMT/blob/main/scripts/reconstruction_ZINC.sh) for the detailed experimental setup on each dataset).
* First, and second arguments denote the gpu_id and experiment_number.

```sh
sh ./scripts/reconstruction_ZINC.sh 0 000
```

* Graph Reconstruction on the synthetic datasets, including grid and ring graphs (See the [script file](https://github.com/JinheonBaek/GMT/blob/main/scripts/reconstruction_synthetic.sh) for the detailed experimental setup on each dataset).
* First, and second arguments denote the gpu_id and experiment_number.

```sh
sh ./scripts/reconstruction_synthetic.sh 0 000
```

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work. </br>

```BibTex
@inproceedings{
    baek2021accurate,
    title={Accurate Learning of Graph Representations with Graph Multiset Pooling},
    author={Jinheon Baek and Minki Kang and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=JHcqXGaqiGn}
}
```
