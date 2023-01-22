# Altegrad Challenge 2022-2023

**Authors: Hugo DEBES, Rita-Mathilda KABRO, Vincent TCHOUMBA**

This project is part of the Advanced Learning for Text and Graph Data (Altegrad) course given at Ecole Polytechnique.

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/hugodebes/Altegrad">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Cellular Component Ontology Prediction</h3>

  <p align="center">
    Bio-informatics competition
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## About The Project

The goal of this project is to study and apply machine learning/artificial intelligence techniques to a classification problem from the field of bio-informatics. Machine learning for protein engineering has attracted a lot of attention recently. Proteins are large biomolecules and macromolecules composed of one or more long chains of amino acids, and are an essential part of all living organisms. Among other, they enable chemical reactions to occur in cells by acting as enzymes and promoting specific reactions, and also provide structural support and play a key role in the immune system’s ability to distinguish self from invaders. They consist of small molecules called amino acids, with long proteins containing up to 4, 500 of these amino acids. There are 20 different amino acids commonly found in the proteins of living organisms. When amino acids bind together, they form a long chain called a polypeptide that can be represented by the protein sequence. The sequence of amino acids then begins to fold, creating the 3D shape of the protein. This structure determines its specific chemical functionality, but the exact details of this process are not yet fully understood. In this challenge, you are given the sequence of 6, 111 proteins along with the graph representation of their structure. The nodes of these graphs represent the amino acids and two nodes are connected by an edge based on the Euclidean distance between these residues, their order in the sequence, and the different chemical interactions between them. The goal of this challenge is to use the sequence and structure of those proteins and classify them into 18 different classes, each representing a characteristic of the location where the protein performs its function obtained from the Cellular Component ontology.

## Organisation

We've tested several approaches in order to achieve the lowest loss possible. You can find the
baseline model written by the teachers as a start in the **baseline** folder.

The **scripts** folder contains Python Scripts to run our data pipeline from **feature_extraction** to building **deep_learning** or **machine_learning** models. Using the sequence of the proteins, we developped an LSTM and Attention-based model. With the embeddings form the ProtBert model, we developped a Graph Convolution Networkn Graph Attention Network, SVC and an Light-GBM model. The best results were achieved with the SVC.

All you need to do is execute the [__struct_main.py__, __seq_main.py__, __ml_main.py__] files to run and evaluate the architectures.

A complete report is avalaible (**report.pdf**) regarding the project, our difficulties and ways to overcome them.

You'll find in the **sandbox** folder some notebooks where we explored solutions that were not conclusive.

If you have any questions, You can contact us at : first_name.last_name@polytechnique.edu

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/hugodebes/Altegrad
   ```
2. Install the project
   ```sh
   pip install -r requirements.txt
   ```
