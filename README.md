## GNN
Graph neural network/Graph embedding code.  
* GNN: gcn, grageSAGE
* GE: node2vec

## GCN
GCN: learn a function of signals/features on a graph ![gnn](http://latex.codecogs.com/png.latex?G=\left(V,E\right))  which takes as input:  
* A feature description xi for every node i; summarized in a N×D feature matrix X (N: number of nodes, D: number of input features)
* A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix A (or some function thereof)and produces a node-level output Z (an N×F feature matrix, where F is the number of output features per node). 

Every neural network layer can then be written as a non-linear function, consider the following simple form of a layer-wise propagation rule:  

![gnn](http://latex.codecogs.com/png.latex?H^{(l+1)}=f\left(H^{(l)},A\right)=\sigma\left(AH^{(l)}W^{(l)}\right))  

with H(0)=X and H(L)=Z (or z for graph-level outputs), L being the number of layers. The specific models then differ only in how f(⋅,⋅) is chosen and parameterized.

## Acknowledgements
The original version of this code base was originally forked from   

**Semi-Supervised Classification with Graph Convolutional Networks**  
https://github.com/tkipf/gcn/ 

**Inductive Representation Learning on Large Graphs**  
https://github.com/williamleif/graphsage-simple

**node2vec: Scalable Feature Learning for Networks**  
https://github.com/aditya-grover/node2vec