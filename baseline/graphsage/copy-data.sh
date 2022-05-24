mkdir /local/mag240m/
cp ~/datasets/dgl-data/mag240m/paper_feat.* ~/datasets/dgl-data/mag240m/graph_bidirected.dgl /local/mag240m/
ln -s /local/mag240m/paper_feat.feat /local/mag240m/feat.feat
ln -s /local/mag240m/paper_feat.shape /local/mag240m/feat.shape
ln -s /local/mag240m/graph_bidirected.dgl /local/mag240m/graph.dgl
