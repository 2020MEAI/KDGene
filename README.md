# KDGene: Knowledge Graph Completion for Disease Gene Prediction using Interactional Tensor Decomposition
### Overview

![KDGeneFramework](KDGeneFramework.png)

We integrate multiple relations centered on diseases and genes from biological knowledge bases to construct a large-scale biological KG, and develop an end-to-end knowledge graph completion model using an interactional tensor decomposition to identify disease-gene associations, named KDGene. 

KDGene introduces a gating mechanism-based interaction module between the embeddings of entities and relations to tensor decomposition, which can effectively enhance the information interaction in biological knowledge. Perceiving related knowledge, the model is capable of learning the connotation of different relations and endows biological entities and relations with more comprehensive and precise representations, which is beneficial to disease gene prediction. 

Experimental results show that KDGene significantly outperforms conventional algorithms, whether existing disease gene prediction methods or knowledge graph embedding methods for general domains. Furthermore, the comprehensive biological analysis of the case of predicted results confirms KDGene's ability to identify new and accurate candidate genes.

### Requirements

Before you begin, ensure you have the following packages on your system:

- python 3.7
- torch 1.7
- numpy

### Usage

```
python learn.py
```

The current hyperparameter default is the same setting we use in this paper (for all ten folds).

To run the KDGene model, several parameters can be adjusted according to your specific needs. Below is a list of these parameters along with their descriptions: 

- `--edim`: (int) Entity embedding dimension. 
- `--rdim`: (int) Relation embedding dimension. 
- `--batch_size`: (int) Size of the batch used in training.
- `--learning_rate`: (float) Learning rate used for optimization. 
- `--regFlag`: (bool) Whether to use N3 regularizer. 
- `--reg`: (float) N3 Regularization coefficient. 
- `--device`: (str) use CPU or GPU.
- `--fold`: (int) from 1 to 10, ten-fold cross-validation.

Here is an example command to run the KDGene model with specified parameters: 

```bash
python learn.py --edim 1000 --rdim 1000 --batch_size 512 --learning_rate 0.1 --reg 0.1 --device 'cuda:0' --fold 1
```

Adjust these parameters based on your dataset and the computational resources available to you.

#### demo 

Additionally, we provide a demo version of the dataset for debugging purposes. This demo consists of ultra-small training and testing sets, along with a reduced version of the knowledge graph. To use this demo dataset, set the `fold` parameter to `"demo"` and update the `base_kg` in `datasets.py` at line 66 to `['demo']`. Then, execute `learn.py` to run the demonstration. Please note, this dataset is intended solely for demonstration purposes and should not be used for result evaluation.

### Dataset

We select curated disease-gene associations from the DisGeNet database [1] as a benchmark dataset and apply the 10-fold cross-validation to evaluate the disease gene prediction algorithms. For each fold, there are 117,738 disease-gene associations in the training set and 13,082 in the testing set.  The dataset can be accessed at the following path: `/KDGene/DisGeNet_cv`.

To learn more comprehensive representations of diseases and genes, we introduce the knowledge of different relation types to construct a biological KG. Regarding diseases, the Disease-Symptom relations from SymMap [2] and the Drug-Disease relations from SIDER [3] are introduced into the KG. Regarding genes, we introduce the Protein-Protein Interactions (PPI) from STRING [4], the Drug-Protein relations from STITCH [5], the Protein-GO relations from GO [6], and the Protein-Pathway relations from KEGG [7]. All triples in the knowledge graph can be accessed at the path: `/KDGene/src_data`. **When opting to incorporate various types of relations into the knowledge graph**, it's essential to update the corresponding section within the code. Specifically, this adjustment needs to be made at line 66 in the datasets.py file.

In our framework, there are no restrictions on the entity type and relation type which means the construction of the KG is flexible. When you use it, the disease-gene relation facts can be added or subtracted from the KG to complete the training according to the demand. As the amount of knowledge in biomedical databases grows, abundant facts about new types of relations can be continuously added to the KG. **It is important to note** that when incorporating additional data, it is necessary to update the lists of entity and relation IDs. These lists can be found and updated in the file located at `/KDGene/DisGeNet_cv/ent_id (or rel_id)`.

### Reference

[1] Piñero J, Bravo À, Queralt-Rosinach N, et al. DisGeNET: a comprehensive platform integrating information on human disease-associated genes and variants[J]. Nucleic acids research, 2016: gkw943.

[2] Wu Y, Zhang F, Yang K, et al. SymMap: an integrative database of traditional Chinese medicine enhanced by symptom mapping[J]. Nucleic acids research, 2019, 47(D1): D1110-D1117.

[3] Kuhn M, Letunic I, Jensen L J, et al. The SIDER database of drugs and side effects[J]. Nucleic acids research, 2016, 44(D1): D1075-D1079.

[4] Von Mering C, Jensen L J, Snel B, et al. STRING: known and predicted protein–protein associations, integrated and transferred across organisms[J]. Nucleic acids research, 2005, 33(suppl_1): D433-D437.

[5] Szklarczyk D, Santos A, Von Mering C, et al. STITCH 5: augmenting protein–chemical interaction networks with tissue and affinity data[J]. Nucleic acids research, 2016, 44(D1): D380-D384.

[6] The Gene Ontology resource: enriching a GOld mine[J]. Nucleic acids research, 2021, 49(D1): D325-D334.

[7] Kanehisa M, Goto S. KEGG: kyoto encyclopedia of genes and genomes[J]. Nucleic acids research, 2000, 28(1): 27-30.

#### Referenced baseline implementations

For the Disease Gene Prediction (DGP) baselines:

- RWRH: https://github.com/hydradon/RwrH

- PRINCE: https://github.com/shub91/Network_Propagation

- DADA: http://compbio.case.edu/dada/

- GUILD: https://github.com/yangkuoone/HerGePred/tree/master/GUILD

- PDGNet: https://github.com/yangkuoone/PDGNet

- GLIM_DG (RWR_PPI/RWR_HMLN): https://github.com/Svvord/GLIM

For the Knowledge Graph Embedding (KGE) baselines used in our study, we have systematically organized and made them available in the GitHub repository:

- Knowledge Graph Embedding (KGE): https://github.com/sienna-wxy/KG_Completion

