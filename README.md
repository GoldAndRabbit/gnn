## GNN
Graph neural network/Graph embedding code.  
* GNN: gcn, grageSAGE
* GE: node2vec

## GCN  
<div style="text-align: center;">
<img alt="" src="https://s3.ax1x.com/2020/11/15/DFYfqe.png" style="display: inline-block;"/>
</div>

GCN: learn a function of signals/features on a graph &nbsp; ![graph](http://latex.codecogs.com/png.latex?G=\left(V,E\right))  &nbsp; which takes as input:  
* A feature description ![x_i](http://latex.codecogs.com/png.latex?x_i)  for every node ![i](http://latex.codecogs.com/png.latex?i) summarized in a ![N×D](http://latex.codecogs.com/png.latex?N\times{D}) feature matrix ![X](http://latex.codecogs.com/png.latex?X) (![N](http://latex.codecogs.com/png.latex?N) : number of nodes, ![D](http://latex.codecogs.com/png.latex?D): number of input features)
* A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix ![A](http://latex.codecogs.com/png.latex?A) and produces a node-level output ![Z](http://latex.codecogs.com/png.latex?Z) (an ![N×F](http://latex.codecogs.com/png.latex?N\times{F}) feature matrix, where ![F](http://latex.codecogs.com/png.latex?F) is the number of output features per node). 

Every neural network layer can then be written as a non-linear function, consider the following simple form of a layer-wise propagation rule:  

![gnn](http://latex.codecogs.com/png.latex?H^{(l+1)}=f\left(H^{(l)},A\right)=\sigma\left(AH^{(l)}W^{(l)}\right))  

with &nbsp; ![H(0)=X](http://latex.codecogs.com/png.latex?H\left(0\right)=X) &nbsp; and &nbsp; ![H(L)=Z](http://latex.codecogs.com/png.latex?H\left(L\right)=Z) &nbsp; (or ![z](http://latex.codecogs.com/png.latex?z) for graph-level outputs), ![L](http://latex.codecogs.com/png.latex?L) being the number of layers. The specific models then differ only in how ![f(⋅,⋅)](http://latex.codecogs.com/png.latex?f\left(\cdot,{\cdot}\right))  is chosen and parameterized.

## Acknowledgements
The original version of this code base was originally forked from   

**Semi-Supervised Classification with Graph Convolutional Networks**  
https://github.com/tkipf/gcn/ 

**Inductive Representation Learning on Large Graphs**  
https://github.com/williamleif/graphsage-simple

**node2vec: Scalable Feature Learning for Networks**  
https://github.com/aditya-grover/node2vec