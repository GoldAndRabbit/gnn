## GNN
Graph neural network/Graph embedding code.  
* GNN: gcn, graphSAGE
*  GE: node2vec

## GCN  

<div style="text-align: center;">
<img alt="" src="https://s3.ax1x.com/2020/11/15/DFYfqe.png" style="display: inline-block;" width="666"/>
</div>

GCN: learn a function of signals/features on a graph _G = (V, E)_ which takes as input:  

* A feature description _x<sub>i</sub>_ for every node i summarized in a _N x D_ feature matrix _X_. (_N_ : number of nodes, _D_ : number of input features)
* A representative description of the graph structure in matrix form, eg: adjacency matrix _A_.

and produces a node-level output _Z_ (an _N x F_ feature matrix, where _F_ is the number of output features per node). Graph-level outputs can be modeled by producing some form of pooling operation.

Every neural network layer can then be written as a non-linear function, consider the following simple form of a layer-wise propagation rule:  

![gnn](http://latex.codecogs.com/png.latex?H^{(l+1)}=f\left(H^{(l)},A\right)=\sigma\left(AH^{(l)}W^{(l)}\right))  

with _H<sup>(0)</sup>=X_ and _H<sup>(L)</sup>=Z_ (_z_ for graph-level outputs), _L_ being the number of layers. The specific models then differ only in how ![f(⋅,⋅)](http://latex.codecogs.com/png.latex?f\left(\cdot,{\cdot}\right))  is chosen and parameterized.


## Preference 
* Semi-Supervised Classification with Graph Convolutional Networks 
* Inductive Representation Learning on Large Graphs  
* node2vec: Scalable Feature Learning for Networks  
