## Node Feature Extractors
Our DocGCN proposed four aspect features for Layout Analyzing. We provide the code for extracting density and syntactic features in this directory. For appearance and semantic features, we provide an example on the FUNSD dataset to show how to extract via google colab notebook directly. We also provide the notebooks for extracting [apprearance]() and [semantic]() features separately. 

## Edge Feature Generation
We use two types of edge features to represent the relationship between different segments. Firstly, we use the [gap distance] as the edge weights of Appearance and Density graphs for learning the spatial relations between segments. Additionally, we use the parent-child relations between segments as the edge weights of semantic and syntactic graphs for learning the logical/structural connections between segments. FUNSD provides the parent-child relation annotations. For PubLayNet and Docbank datasets, we can easily use their source code (XML, Latex) by text matching and a reading order assigning a method to extract parent-child relations for the other two datasets. After we got the parent-child relation of the training set, we designed a transformer-based model to predict the parent-child relation for the test set. 

We provide an example of the FUNSD dataset to show how it works. 
<p align="center"><img src="doc_gcn/figures/RDM.png" width="750" /></p>
