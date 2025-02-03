# AI Safety Landscape

Goal: create a visualization of AI Safety research papers using an unsupervised clustering and labeling approach.

## Overview

I was trying to understand how all the different research areas in AI Safety fit together, and I came across the *Future of Life Institute*'s [AI Safety Landscape](https://futureoflife.org/valuealignmentmap/). It's a great visualization, but unfortunately it's quite out of date as it was published in 2017 and hasn't been updated since. Furthermore, it seems it was created manually, but I don't have the expertise to replicate that, and I think given the expansion of the field since, it would be great to have an automated, scalable approach. 
This led me to a tutorial from the *Programming Historian*, [Clustering and Visualising Documents using Word Embeddings](https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings), where the authors uses word embeddings and clustering algorithms to identify groups of similar documents in a corpus of approximately 9,000 academic abstracts. However, they rely on the Dewey Decimal Classification (DDC) as ground truth labels, which are very coarse-grained and not suitable for this use case.
My idea was thus to combine the embedding and clustering approach, but use an LLM to label the clusters.

## Pipeline

### Phase 1: Data Collection

I decided to focus solely on arXiv papers, as they provide an easy to use [OAI-PMH endpoint](https://info.arxiv.org/help/oa/index.html) for [bulk metadata harvest](https://info.arxiv.org/help/bulk_data.html) and include many papers in the field. They provide two metadata formats, `arXiv` and `arXivRaw`. Ideally, `arXiv` would be sufficient, as it has the advantage of separating out authors, but they don't clearly mark withdrawn papers, so I also harvested the `arXivRaw` metadata which includes all the versioning information, which can be used to infer their withdrawn status. I could have probably skipped this step, however at the time I thought I might use the full paper text (also available with public APIs as a TeX source), but eventually dropped that idea. At this stage, I also normalize the authors' names (there are some inconsistencies in suffixes such as Jr. and Sr., for example). All the data is stored in a PostgreSQL database. At the time of collection, there were 712,724 CS papers in the arXiv database. 

### Phase 2: Embedding Generation

Based on the [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard), `voyage-3-m-exp` is at the top of the ArxivClusteringP2P task, which aims to cluster papers based on their title and abstracts to assign them an arXiv category. While we aim for much more granularity, this is a good starting point, and the voyage embeddings are very high quality in general. The [`voyage-3-m-exp` page](https://huggingface.co/voyageai/voyage-3-m-exp) says that "voyage-3-large is highly recommended and likey strictly better than voyage-3-m-exp", so I selected the `voyage-3-large` model, which can output 2048-dimensional embeddings. At the time of writing, Voyage AI offer [200M free tokens](https://docs.voyageai.com/docs/pricing#text-embeddings), which was more than enough for this project. At this stage, I also wanted to start filtering papers, as 700K was probably too much and I was only really interested in AI safety papers. Thus, I only included papers with `cs.AI` as one of their categories. I only realized later that **this was a major mistake**, as many AI safety papers are not classified under `cs.AI`, but rather under `cs.LG` ('Machine Learning'). For example, the ["AI Control: Improving Safety Despite Intentional Subversion"](https://arxiv.org/abs/2312.06942) paper is classified solely under `cs.LG`. In any case, there were 112,313 `cs.AI` papers, which resulted in 28,677,566 tokens for when embedding their concatenated title and abstract.

> I initially tried to use [ModernBERT](https://huggingface.co/blog/modernbert), but the resulting embeddings were of much lower quality: the range of similarities between papers was much smaller. I suspect it hasn't been optimized for a downstream clustering task as much. 

### Phase 3: Clustering

I selected [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) (Hierarchical Density-Based Spatial Clustering of Applications with Noise) as a clustering algorithm. It has several advantages:
- **It can provide a hierarchical clustering**, which would theoretically allow us to create a taxonomy of AI safety research, mirroring the FLI's AI Safety Landscape.
- There is no need to specify the number of clusters beforehand.
- Clusters can be of varying densities, accounting for mainstream or niche research areas naturally.
- It has built-in noise detection.
- It excels at clustering high-dimensional data.
- Considered SOTA in for many clustering tasks.

#### Dimensionality Reduction with UMAP

To improve accuracy of clustering tasks, a dimensionality reduction is often applied as a preprocessing step, as HDBSCAN is still sensitive to the curse of dimensionality. I used [UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection), as it's fast, preserves the local and global structure of the data, and is generally well-suited to a downstream HDBSCAN clustering.

#### Hyperparameter Tuning

I used [optuna](https://optuna.readthedocs.io/en/stable/) for hyperparameter optimization. It allows multiprocessing and can connect to a database backend, which was helpful for resuming tasks. Using the tutorials from [HDBSCAN's documentation](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) and UMAP ([1](https://umap-learn.readthedocs.io/en/latest/parameters.html) and [2](https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering)), I selected the following hyperparameters to optimize:

- `use_umap` (range: `[True, False]`): whether to use UMAP for dimensionality reduction. I was curious to see if it would improve the clustering results, as HDBSCAN can supposedly handle high-dimensional data (*indeed, UMAP reduction helped*)
- `min_cluster_size` (range: `[20, 100]`): the minimum number of points in a cluster, this is the primary parameter for HDBSCAN.
- `min_samples` (range: `[5, 50]`): provides a measure of how conservative the clustering is, the higher it is, the more points will be considered noise.
- `cluster_selection_epsilon` (range: `[0.0, 0.5]`): allows merging of micro-clusters when min_cluster_size is low. 
- `n_components` (range: `[15, 100]`): the number of dimensions in the UMAP embedding.
- `n_neighbors` (range: `[30, 100]`): the number of neighbors to use in the UMAP embedding.

The following hyperparameters were kept constant:
- `metric = 'cosine'`: the metric to use for the UMAP embedding. `cosine` is standard for text embeddings/semantic similarity.
- `min_dist = 0.0`: the minimum distance between points in the UMAP embedding. As per the tutorial, for a downstream clustering task, it is best to set this to 0.0.
- `cluster_selection_method='leaf'`: the method to use for cluster selection. `eom` (Excess of Mass) is the default, but it has a tendency to pick one or two large clusters and then a number of small extra clusters. Instead, a better option is to select 'leaf' as a cluster selection method. This will select leaf nodes from the tree, producing many small homogeneous clusters, wnich is a better fit for granular research areas.

#### Objective Function

For evaluating cluster quality, I selected the Density-based Clustering Validation Index (DBCVI) as the optimization objective, which as the name implies is specifically designed for density-based clustering. More precisely, `hdbscan` provides a `_relative_validity` attribute that is a fast approximation suitable for comparing results. It had to be reimplemented to work with cuML's GPU library but the effort was minimal with the source code. I rejected other metrics often used for clustering tasks, like Ball Hall, Davies Bouldin, Calinski Harabasz, Silhouette, and R-squared indices as they make assumptions such as spherical clusters or equal densities that I wasn't sure would align with the semantic structure of academic papers. I also considered 'trustworthiness' as a metric specifically for evaluating UMAP reduction quality, but given that UMAP quality by itself was not the objective, and that I wanted to verify that using was actually better, I only saved it as a metric and did not use it for optimization.

#### Optimization Results

The best trial found was # 451, with a relative validity of 0.294. The parameters used were:
```json
{
    "use_umap": true,
    "n_components": 37,
    "n_neighbors": 48,
    "min_cluster_size": 96,
    "min_samples": 21,
    "cluster_selection_epsilon": 0.166
}
```

> Given that the min_cluster_size is so close the the end of the range, it suggests that a better clustering could have been found with an expanded range. 

#### Clustering Results

HDBSCAN identified 153 clusters, which is about expected given all `cs.AI` papers were clustered, and a high-granularity was sought. However, a relatively high noise+ ratio of 49.57% indicates that about half of the papers couldn't be confidently assigned to any cluster. While this could mean either emerging research areas or papers that bridge multiple domains, it's more likely that further work should be done on the clustering parameters. The clusters also show considerable variation in size (standard deviation of 402 papers) with a size ratio of 29.6 between the largest and smallest clusters. This could be a mix of the presence of both mainstream research areas and more specialized niches, as well as insufficiently granular clustering. The average cluster probability of 0.43 (±0.45) indicates moderate confidence in the cluster assignments, while the mean persistence of 0.125 suggests the clusters are reasonably stable. The trust score of 0.66 for the UMAP dimensionality reduction demonstrates that the lower-dimensional representation preserved a good portion of the original semantic relationships between papers. The relative validity of 0.2943 cannot be used as a measure of the quality of the clustering directly, and a full score needs to be computed.

> I had issues calculating the full DBCVI score. I suspect there was some bug in how the data was stored in the database, and a new clustering might be needed.

### Phase 4: Labeling

I used Gemini 1.5 Flash with structured output to label the clusters. I also took the opportunity to add a relevance score to the labels, so that labels could then be filtered for a final visualization. The prompt used was:

```
You are an expert in AI safety and machine learning. Your task is to generate precise technical labels for clusters of academic papers related to AI research.

I will provide the ten papers most representative of the cluster (closest to the cluster centroid).

Review these papers and provide:
1. A specific technical category that precisely describes the research area represented by this cluster
2. A relevance score (0-1) indicating how relevant this research area is to AI safety

Guidelines:
- Use precise technical terminology
- Categories should be specific enough to differentiate between related research areas yet broad enough to actually group papers (e.g. "Reward Modeling for RLHF" rather than "Reinforcement Learning" or "Regularizing Hidden States Enables Learning Generalizable Reward Model for RLHF")
- Consider both direct and indirect relevance to AI safety

Papers to analyze:
```
Then, a batch of 10 representative papers, with titles and abstracts, were provided. These papers chosen were those closest to the cluster centroid.

#### Labeling Results

##### Clusters by Size

| Size | Label | Safety Relevance |
|------|-------|------------------|
| 2844 | Multiagent Learning in Repeated Games and Imperfect Information Settings |       0.70       |
| 2262 | Explainable AI (XAI) for Regression and Classification Models |       0.80       |
| 2221 | Probabilistic Conditional Independence in Graphical Models |       0.20       |
| 1593 | Game-Theoretic Motion Planning for Autonomous Vehicles in Dense Traffic |       0.90       |
| 1375 | Federated Learning Frameworks and Optimization |       0.70       |
| 1368 | Graph Neural Network Enhancement for Heterophilic Graphs |       0.10       |
| 1159 | Multitask Finetuning for Foundation Model Adaptation |       0.60       |
| 1106 | Algorithmic Fairness and Bias Mitigation |       0.80       |
| 1092 | Adversarial Robustness in Deep Learning |       0.90       |
| 1075 | Neural Network Optimization and Generalization |       0.20       |
| 927  | Continual Lifelong Learning |       0.50       |
| 875  | Deep Neural Network Compression Techniques |       0.20       |
| 797  | Data Collection and Processing Techniques for Virtual and Biological Systems |       0.20       |
| 743  | Ontology Reasoning and Knowledge Representation |       0.20       |
| 731  | Large Language Model (LLM) based Program Repair and Explainability |       0.70       |
| 729  | Knowledge Graph Embedding for Link Prediction |       0.20       |
| 669  | Unsupervised Domain Adaptation and Generalization |       0.20       |
| 657  | Anomaly Detection Techniques for Heterogeneous and Complex Data |       0.60       |
| 632  | Machine Learning for Combinatorial Optimization |       0.20       |
| 631  | Uncertainty Quantification in Neural Networks |       0.80       |
| 585  | Domain Adaptation Techniques for Neural Machine Translation |       0.10       |
| 578  | Off-Policy Deep Reinforcement Learning Algorithms and Convergence |       0.70       |
| 573  | Multimodal Emotion Recognition |       0.20       |
| 571  | Deep Reinforcement Learning for Dexterous Robotic Manipulation |       0.20       |
| 570  | Deep Reinforcement Learning for Wireless Resource Management |       0.20       |
| 554  | Video Understanding and Representation Learning |       0.20       |
| 550  | Text-to-Text Frameworks for Recommendation Systems |       0.20       |
| 519  | Few-Shot Learning using Prototypical Networks and Feature Extraction |       0.20       |
| 516  | Evolutionary Algorithm Enhancements for Multimodal Optimization |       0.20       |
| 514  | AI Ethics and Governance |       0.95       |
| 461  | Data-driven Predictive Modeling for Building Energy Management |       0.20       |
| 454  | Spiking Neural Networks and Novel Training Methods |       0.20       |
| 454  | Reward Learning and Inverse Reinforcement Learning for Safe RL |       0.80       |
| 448  | Multi-Agent Pathfinding and Planning |       0.20       |
| 443  | Multi-hop Question Answering and Question Generation |       0.20       |
| 442  | Unsupervised Misinformation Detection and Rumor Classification on Social Media |       0.80       |
| 417  | Partially Observable Markov Decision Process (POMDP) Solution Methods |       0.20       |
| 409  | Hate Speech and Toxicity Detection in Low-Resource Languages |       0.90       |
| 409  | Visual Question Answering with Graph Neural Networks and Relational Reasoning |       0.20       |
| 405  | Multimodal Soft Sensing and Knowledge Integration using Large Language Models |       0.20       |
| 404  | Knowledge Distillation and Model Compression for Language Models and Neural Networks |       0.20       |
| 401  | Quantum Algorithm Optimization and Architecture Search using Machine Learning |       0.20       |
| 392  | Statistical Methods for Social Choice |       0.20       |
| 390  | Embodied Instruction Following and Navigation |       0.20       |
| 388  | Offline Reinforcement Learning Algorithms and Datasets |       0.70       |
| 388  | Generative 3D Molecule Design for Drug Discovery |       0.70       |
| 383  | Neural Relation Extraction and Debiasing Techniques |       0.20       |
| 376  | Efficient Vision Transformer Architectures for Resource-Constrained Environments |       0.20       |
| 372  | Synthetic Media Detection and Generation |       0.80       |
| 370  | Neural Abstractive Text Summarization |       0.10       |
| 364  | Deep Learning for Software Engineering |       0.20       |
| 340  | Neural Operator Methods for Solving Partial Differential Equations |       0.20       |
| 338  | Medical Image Segmentation using Deep Learning |       0.20       |
| 334  | Safe Reinforcement Learning with Constraint Satisfaction |       1.00       |
| 332  | Music Generation from Lyrics and Audio |       0.10       |
| 331  | Neural Implicit Representations for 3D Scene Reconstruction |       0.10       |
| 331  | Answer Set Programming Semantics and Computation |       0.10       |
| 325  | Philosophical Foundations and Theoretical Models of Artificial Intelligence |       0.20       |
| 322  | Abstract Argumentation Frameworks: Semantics and Expressiveness |       0.20       |
| 317  | EEG-based Brain-Computer Interfaces and Decoding |       0.20       |
| 315  | Contextual Bandit Algorithms and Theory |       0.20       |
| 311  | Spatiotemporal Graph Neural Networks for Traffic Forecasting |       0.10       |
| 309  | Automated Neural Architecture Search (NAS) |       0.20       |
| 308  | Object-Centric Process Mining and Conformance Checking |       0.10       |
| 305  | Automated System Log Analysis and Anomaly Detection |       0.80       |
| 301  | Student Performance Prediction using Machine Learning |       0.10       |
| 296  | Bayesian Optimization Methods and Scalable Algorithms |       0.20       |
| 293  | Self-Supervised Visual Representation Learning |       0.20       |
| 291  | Unsupervised and Transfer Learning for Textual and Visual Embeddings |       0.20       |
| 283  | Financial Time Series Forecasting using Deep Learning |       0.10       |
| 278  | Reinforcement Learning for Legged Robot Locomotion Control |       0.70       |
| 277  | Clinical Representation Learning and Prediction using Electronic Health Records |       0.20       |
| 266  | 3D Human Pose and Shape Estimation from Monocular and Multi-View Data |       0.10       |
| 259  | Online and Incremental Learning for Convolutional Neural Networks |       0.20       |
| 257  | Intrinsically Motivated Exploration in Reinforcement Learning |       0.70       |
| 256  | Active Learning Algorithms and Strategies |       0.20       |
| 247  | Differentiable Inductive Logic Programming |       0.30       |
| 242  | Deep Learning for Image and Video Enhancement |       0.20       |
| 242  | Fuzzy Set Theory and its Applications in Multicriteria Decision Making |       0.20       |
| 240  | ECG-based Cardiovascular Disease Classification using Deep Learning |       0.80       |
| 228  | Backdoor Attacks in Deep Neural Networks |       0.90       |
| 224  | Model Extraction Attacks and Defenses in Machine Learning |       0.80       |
| 210  | Mechanism Design and Revenue Maximization in Online Auctions |       0.20       |
| 208  | Differentially Private Data Synthesis and Generative Models |       0.70       |
| 207  | Explainable Planning and Interactive Human-Robot Collaboration |       0.70       |
| 205  | Large Language Model Jailbreaking and Security |       0.95       |
| 204  | Named Entity Recognition (NER) Techniques and Improvements |       0.10       |
| 202  | Zero-Shot and Minimally Supervised Speech Synthesis |       0.10       |
| 200  | Noisy Label Learning and Robustness in Machine Learning Models |       0.70       |
| 199  | Addressing Class Imbalance in Deep Learning |       0.20       |
| 194  | Hierarchical Reinforcement Learning (HRL) Option Discovery and Skill Acquisition |       0.60       |
| 186  | Document Image Analysis and Optical Character Recognition (OCR) |       0.20       |
| 181  | Time Series Forecasting using Deep Learning |       0.10       |
| 177  | Real-time Object Detection in Aerial and Autonomous Driving Imagery |       0.70       |
| 174  | Graph-based Clustering and Representation Learning |       0.20       |
| 172  | AI-Generated Text Detection and Watermarking |       0.70       |
| 171  | Fine-Grained Controllable Story Generation |       0.20       |
| 171  | Network Intrusion Detection Systems using Machine Learning in IoT and SDN environments |       0.70       |
| 168  | Risk-Sensitive Reinforcement Learning Algorithms |       0.80       |
| 168  | Deep Reinforcement Learning for Mobile Robot Navigation |       0.40       |
| 167  | End-to-End Automatic Speech Recognition (ASR) Model Optimization |       0.10       |
| 164  | Multi-class and Multi-label Classification Techniques |       0.20       |
| 162  | Sequential Pattern Mining Algorithms and Optimizations |       0.20       |
| 161  | Sequential Monte Carlo (SMC) Methods for Probabilistic Programming |       0.20       |
| 156  | Knowledge Graph Entity Alignment |       0.10       |
| 155  | Fairness and Bias Mitigation in Recommender Systems |       0.80       |
| 152  | Adversarial Attacks on Deep Reinforcement Learning |       0.90       |
| 151  | Model-Based Reinforcement Learning Algorithms |       0.30       |
| 151  | Disentangled Representation Learning in Variational Autoencoders |       0.20       |
| 150  | Computational Creativity and Generative Models |       0.20       |
| 149  | Natural Language to SQL Query Generation |       0.20       |
| 149  | Game-Theoretic Modeling of Adversarial Interactions in Cybersecurity |       0.80       |
| 148  | Contrastive Vision-Language Pre-training Improvements |       0.20       |
| 147  | Program Synthesis via Inductive Learning and Neural Methods |       0.20       |
| 140  | Explainable Reinforcement Learning (XRL) |       0.80       |
| 137  | Legal Natural Language Processing (NLP) |       0.20       |
| 137  | Multi-Modal Sensor Fusion for 3D Object Detection in Autonomous Driving |       0.80       |
| 136  | Malware Detection using Deep Learning and Data Augmentation |       0.70       |
| 136  | Fair and Efficient Resource Allocation Mechanisms |       0.20       |
| 131  | Graph Neural Network Explainability Methods |       0.70       |
| 131  | Distributed Deep Learning Training Optimization |       0.10       |
| 126  | Plant Disease Detection using Computer Vision |       0.10       |
| 126  | Generative Adversarial Network (GAN) Training Stability and Generalization |       0.20       |
| 125  | Mathematical Foundations of Rough Set Theory |       0.10       |
| 124  | Multi-Task Learning (MTL) Architectures and Algorithms |       0.20       |
| 124  | Aspect-Based Sentiment Analysis (ABSA) Techniques |       0.10       |
| 124  | Affective Human-Robot Interaction |       0.30       |
| 122  | Human-in-the-loop Video and Text Processing |       0.20       |
| 120  | Quantum Cognition Modeling |       0.20       |
| 119  | Large Language Model (LLM) Behavior Analysis and Limitations |       0.70       |
| 118  | Automated Ophthalmic Disease Detection using Deep Learning |       0.20       |
| 118  | Semi-supervised Contrastive Learning for Medical Image Segmentation |       0.20       |
| 116  | Zero-Shot Learning (ZSL) and Generalized Zero-Shot Learning (GZSL) for Image Classification |       0.20       |
| 115  | Adversarial Attacks and Defenses in Natural Language Processing |       0.80       |
| 113  | Deep Active Inference for Planning and Control |       0.70       |
| 113  | Deep Learning for Surgical Skill Assessment and Training |       0.70       |
| 112  | Click-Through Rate (CTR) Prediction using Deep Learning |       0.10       |
| 111  | StyleGAN Latent Space Manipulation for Image Editing and Synthesis |       0.20       |
| 110  | Real-time Instance Segmentation |       0.20       |
| 109  | Off-Policy Meta-Reinforcement Learning |       0.20       |
| 109  | Transfer Learning in Reinforcement Learning |       0.30       |
| 108  | Compositional Generalization in Seq2Seq Models |       0.20       |
| 107  | Explainable Recommender Systems |       0.70       |
| 104  | Unsupervised and Semi-Supervised Feature Learning for Human Activity Recognition from Wearable Sensors |       0.20       |
| 102  | Agent-Based Modeling for Pandemic Simulation and Intervention |       0.70       |
| 100  | Proceedings of AI Workshops and Conferences |       0.20       |
|  99  | Self-Supervised Monocular Depth Estimation |       0.30       |
|  99  | Job Shop Scheduling Optimization |       0.10       |
|  98  | Visual Attention and Gaze Prediction |       0.20       |
|  98  | AI-assisted Elderly Care: Multimodal Sensor Data Analysis for Activity Monitoring and Fall Detection |       0.70       |
|  98  | Automated Depression Detection using Multimodal Machine Learning |       0.70       |
|  97  | Quality Diversity (QD) Algorithms for Reinforcement Learning and Neuroevolution |       0.70       |
|  96  | Adversarial Attacks on Graph Neural Networks |       0.70       |


##### Clusters by Safety Relevance

| Safety Relevance | Size | Label |
|-------------------|------|-------|
|       1.00        | 334  | Safe Reinforcement Learning with Constraint Satisfaction |
|       0.95        | 205  | Large Language Model Jailbreaking and Security |
|       0.95        | 514  | AI Ethics and Governance |
|       0.90        | 152  | Adversarial Attacks on Deep Reinforcement Learning |
|       0.90        | 1593 | Game-Theoretic Motion Planning for Autonomous Vehicles in Dense Traffic |
|       0.90        | 409  | Hate Speech and Toxicity Detection in Low-Resource Languages |
|       0.90        | 1092 | Adversarial Robustness in Deep Learning |
|       0.90        | 228  | Backdoor Attacks in Deep Neural Networks |
|       0.80        | 631  | Uncertainty Quantification in Neural Networks |
|       0.80        | 372  | Synthetic Media Detection and Generation |
|       0.80        | 140  | Explainable Reinforcement Learning (XRL) |
|       0.80        | 442  | Unsupervised Misinformation Detection and Rumor Classification on Social Media |
|       0.80        | 115  | Adversarial Attacks and Defenses in Natural Language Processing |
|       0.80        | 240  | ECG-based Cardiovascular Disease Classification using Deep Learning |
|       0.80        | 305  | Automated System Log Analysis and Anomaly Detection |
|       0.80        | 2262 | Explainable AI (XAI) for Regression and Classification Models |
|       0.80        | 168  | Risk-Sensitive Reinforcement Learning Algorithms |
|       0.80        | 224  | Model Extraction Attacks and Defenses in Machine Learning |
|       0.80        | 155  | Fairness and Bias Mitigation in Recommender Systems |
|       0.80        | 149  | Game-Theoretic Modeling of Adversarial Interactions in Cybersecurity |
|       0.80        | 454  | Reward Learning and Inverse Reinforcement Learning for Safe RL |
|       0.80        | 1106 | Algorithmic Fairness and Bias Mitigation |
|       0.80        | 137  | Multi-Modal Sensor Fusion for 3D Object Detection in Autonomous Driving |
|       0.70        | 131  | Graph Neural Network Explainability Methods |
|       0.70        | 172  | AI-Generated Text Detection and Watermarking |
|       0.70        | 113  | Deep Active Inference for Planning and Control |
|       0.70        |  98  | AI-assisted Elderly Care: Multimodal Sensor Data Analysis for Activity Monitoring and Fall Detection |
|       0.70        | 102  | Agent-Based Modeling for Pandemic Simulation and Intervention |
|       0.70        | 278  | Reinforcement Learning for Legged Robot Locomotion Control |
|       0.70        | 200  | Noisy Label Learning and Robustness in Machine Learning Models |
|       0.70        | 2844 | Multiagent Learning in Repeated Games and Imperfect Information Settings |
|       0.70        | 388  | Offline Reinforcement Learning Algorithms and Datasets |
|       0.70        | 177  | Real-time Object Detection in Aerial and Autonomous Driving Imagery |
|       0.70        | 388  | Generative 3D Molecule Design for Drug Discovery |
|       0.70        | 113  | Deep Learning for Surgical Skill Assessment and Training |
|       0.70        |  97  | Quality Diversity (QD) Algorithms for Reinforcement Learning and Neuroevolution |
|       0.70        |  96  | Adversarial Attacks on Graph Neural Networks |
|       0.70        | 207  | Explainable Planning and Interactive Human-Robot Collaboration |
|       0.70        | 1375 | Federated Learning Frameworks and Optimization |
|       0.70        | 119  | Large Language Model (LLM) Behavior Analysis and Limitations |
|       0.70        |  98  | Automated Depression Detection using Multimodal Machine Learning |
|       0.70        | 208  | Differentially Private Data Synthesis and Generative Models |
|       0.70        | 107  | Explainable Recommender Systems |
|       0.70        | 731  | Large Language Model (LLM) based Program Repair and Explainability |
|       0.70        | 171  | Network Intrusion Detection Systems using Machine Learning in IoT and SDN environments |
|       0.70        | 136  | Malware Detection using Deep Learning and Data Augmentation |
|       0.70        | 578  | Off-Policy Deep Reinforcement Learning Algorithms and Convergence |
|       0.70        | 257  | Intrinsically Motivated Exploration in Reinforcement Learning |
|       0.60        | 657  | Anomaly Detection Techniques for Heterogeneous and Complex Data |
|       0.60        | 1159 | Multitask Finetuning for Foundation Model Adaptation |
|       0.60        | 194  | Hierarchical Reinforcement Learning (HRL) Option Discovery and Skill Acquisition |
|       0.50        | 927  | Continual Lifelong Learning |
|       0.40        | 168  | Deep Reinforcement Learning for Mobile Robot Navigation |
|       0.30        | 247  | Differentiable Inductive Logic Programming |
|       0.30        | 124  | Affective Human-Robot Interaction |
|       0.30        | 151  | Model-Based Reinforcement Learning Algorithms |
|       0.30        | 109  | Transfer Learning in Reinforcement Learning |
|       0.30        |  99  | Self-Supervised Monocular Depth Estimation |
|       0.20        | 116  | Zero-Shot Learning (ZSL) and Generalized Zero-Shot Learning (GZSL) for Image Classification |
|       0.20        | 256  | Active Learning Algorithms and Strategies |
|       0.20        | 124  | Multi-Task Learning (MTL) Architectures and Algorithms |
|       0.20        | 137  | Legal Natural Language Processing (NLP) |
|       0.20        | 364  | Deep Learning for Software Engineering |
|       0.20        | 242  | Deep Learning for Image and Video Enhancement |
|       0.20        |  98  | Visual Attention and Gaze Prediction |
|       0.20        | 147  | Program Synthesis via Inductive Learning and Neural Methods |
|       0.20        | 340  | Neural Operator Methods for Solving Partial Differential Equations |
|       0.20        | 519  | Few-Shot Learning using Prototypical Networks and Feature Extraction |
|       0.20        | 401  | Quantum Algorithm Optimization and Architecture Search using Machine Learning |
|       0.20        | 108  | Compositional Generalization in Seq2Seq Models |
|       0.20        | 461  | Data-driven Predictive Modeling for Building Energy Management |
|       0.20        | 149  | Natural Language to SQL Query Generation |
|       0.20        | 199  | Addressing Class Imbalance in Deep Learning |
|       0.20        | 1075 | Neural Network Optimization and Generalization |
|       0.20        | 122  | Human-in-the-loop Video and Text Processing |
|       0.20        | 186  | Document Image Analysis and Optical Character Recognition (OCR) |
|       0.20        | 164  | Multi-class and Multi-label Classification Techniques |
|       0.20        | 376  | Efficient Vision Transformer Architectures for Resource-Constrained Environments |
|       0.20        | 162  | Sequential Pattern Mining Algorithms and Optimizations |
|       0.20        | 171  | Fine-Grained Controllable Story Generation |
|       0.20        | 118  | Automated Ophthalmic Disease Detection using Deep Learning |
|       0.20        | 669  | Unsupervised Domain Adaptation and Generalization |
|       0.20        | 277  | Clinical Representation Learning and Prediction using Electronic Health Records |
|       0.20        | 404  | Knowledge Distillation and Model Compression for Language Models and Neural Networks |
|       0.20        | 291  | Unsupervised and Transfer Learning for Textual and Visual Embeddings |
|       0.20        | 120  | Quantum Cognition Modeling |
|       0.20        | 210  | Mechanism Design and Revenue Maximization in Online Auctions |
|       0.20        | 161  | Sequential Monte Carlo (SMC) Methods for Probabilistic Programming |
|       0.20        | 317  | EEG-based Brain-Computer Interfaces and Decoding |
|       0.20        | 454  | Spiking Neural Networks and Novel Training Methods |
|       0.20        | 570  | Deep Reinforcement Learning for Wireless Resource Management |
|       0.20        | 174  | Graph-based Clustering and Representation Learning |
|       0.20        | 554  | Video Understanding and Representation Learning |
|       0.20        | 296  | Bayesian Optimization Methods and Scalable Algorithms |
|       0.20        | 293  | Self-Supervised Visual Representation Learning |
|       0.20        | 104  | Unsupervised and Semi-Supervised Feature Learning for Human Activity Recognition from Wearable Sensors |
|       0.20        | 100  | Proceedings of AI Workshops and Conferences |
|       0.20        | 151  | Disentangled Representation Learning in Variational Autoencoders |
|       0.20        | 417  | Partially Observable Markov Decision Process (POMDP) Solution Methods |
|       0.20        | 550  | Text-to-Text Frameworks for Recommendation Systems |
|       0.20        | 383  | Neural Relation Extraction and Debiasing Techniques |
|       0.20        | 110  | Real-time Instance Segmentation |
|       0.20        | 573  | Multimodal Emotion Recognition |
|       0.20        | 315  | Contextual Bandit Algorithms and Theory |
|       0.20        | 325  | Philosophical Foundations and Theoretical Models of Artificial Intelligence |
|       0.20        | 309  | Automated Neural Architecture Search (NAS) |
|       0.20        | 875  | Deep Neural Network Compression Techniques |
|       0.20        | 148  | Contrastive Vision-Language Pre-training Improvements |
|       0.20        | 409  | Visual Question Answering with Graph Neural Networks and Relational Reasoning |
|       0.20        | 390  | Embodied Instruction Following and Navigation |
|       0.20        | 109  | Off-Policy Meta-Reinforcement Learning |
|       0.20        | 516  | Evolutionary Algorithm Enhancements for Multimodal Optimization |
|       0.20        | 448  | Multi-Agent Pathfinding and Planning |
|       0.20        | 571  | Deep Reinforcement Learning for Dexterous Robotic Manipulation |
|       0.20        | 743  | Ontology Reasoning and Knowledge Representation |
|       0.20        | 136  | Fair and Efficient Resource Allocation Mechanisms |
|       0.20        | 126  | Generative Adversarial Network (GAN) Training Stability and Generalization |
|       0.20        | 322  | Abstract Argumentation Frameworks: Semantics and Expressiveness |
|       0.20        | 111  | StyleGAN Latent Space Manipulation for Image Editing and Synthesis |
|       0.20        | 150  | Computational Creativity and Generative Models |
|       0.20        | 392  | Statistical Methods for Social Choice |
|       0.20        | 242  | Fuzzy Set Theory and its Applications in Multicriteria Decision Making |
|       0.20        | 632  | Machine Learning for Combinatorial Optimization |
|       0.20        | 405  | Multimodal Soft Sensing and Knowledge Integration using Large Language Models |
|       0.20        | 338  | Medical Image Segmentation using Deep Learning |
|       0.20        | 118  | Semi-supervised Contrastive Learning for Medical Image Segmentation |
|       0.20        | 2221 | Probabilistic Conditional Independence in Graphical Models |
|       0.20        | 797  | Data Collection and Processing Techniques for Virtual and Biological Systems |
|       0.20        | 259  | Online and Incremental Learning for Convolutional Neural Networks |
|       0.20        | 443  | Multi-hop Question Answering and Question Generation |
|       0.20        | 729  | Knowledge Graph Embedding for Link Prediction |
|       0.10        | 301  | Student Performance Prediction using Machine Learning |
|       0.10        | 283  | Financial Time Series Forecasting using Deep Learning |
|       0.10        | 370  | Neural Abstractive Text Summarization |
|       0.10        | 308  | Object-Centric Process Mining and Conformance Checking |
|       0.10        | 126  | Plant Disease Detection using Computer Vision |
|       0.10        | 124  | Aspect-Based Sentiment Analysis (ABSA) Techniques |
|       0.10        | 1368 | Graph Neural Network Enhancement for Heterophilic Graphs |
|       0.10        | 266  | 3D Human Pose and Shape Estimation from Monocular and Multi-View Data |
|       0.10        | 131  | Distributed Deep Learning Training Optimization |
|       0.10        | 112  | Click-Through Rate (CTR) Prediction using Deep Learning |
|       0.10        | 125  | Mathematical Foundations of Rough Set Theory |
|       0.10        | 332  | Music Generation from Lyrics and Audio |
|       0.10        | 204  | Named Entity Recognition (NER) Techniques and Improvements |
|       0.10        | 311  | Spatiotemporal Graph Neural Networks for Traffic Forecasting |
|       0.10        | 181  | Time Series Forecasting using Deep Learning |
|       0.10        | 202  | Zero-Shot and Minimally Supervised Speech Synthesis |
|       0.10        | 156  | Knowledge Graph Entity Alignment |
|       0.10        |  99  | Job Shop Scheduling Optimization |
|       0.10        | 585  | Domain Adaptation Techniques for Neural Machine Translation |
|       0.10        | 331  | Neural Implicit Representations for 3D Scene Reconstruction |
|       0.10        | 167  | End-to-End Automatic Speech Recognition (ASR) Model Optimization |
|       0.10        | 331  | Answer Set Programming Semantics and Computation |