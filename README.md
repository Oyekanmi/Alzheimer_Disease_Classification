# Multiclass Classification of Alzheimer's Disease Prodromal Stages

This repository provides the implementation of the methods described in the paper **"Multiclass classification of Alzheimer's disease prodromal stages using sequential feature embeddings and regularized multikernel support vector machine"**. The study introduces an advanced framework for distinguishing between the prodromal stages of Alzheimer's disease (AD), namely Cognitive Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD), using a unified, multimodal machine learning approach.

## Research Significance

Alzheimer's disease remains a major challenge due to its progressive nature and the heterogeneity of its prodromal stages, particularly in the MCI category. Existing approaches often rely on sequential binary classification tasks or direct concatenation of features, which are limited in interpretability and prone to overfitting due to feature imbalance. This study addresses these challenges by proposing a robust, unified multiclass classification framework that leverages both neuroimaging and non-imaging biomarkers.

## Methodological Novelty

1. **Sequential Feature Embeddings**:
   - Introduces an **Ensemble Manifold Regularized Sparse Low-Rank Approximation (EMR-SLRA)** for dimensionality reduction while preserving the intrinsic geometry of multimodal data. 
   - This method creates a joint low-dimensional embedding from MRI and PET features, capturing interdependencies between modalities and mitigating the curse of dimensionality.

2. **Regularized Multikernel Support Vector Machine (SVM)**:
   - A novel regularization approach ensures balanced contributions from all data modalities, addressing the common issue of dominance by high-dimensional feature sets.
   - This framework enhances model robustness by integrating neuroimaging (MRI, PET) with genetic and cognitive biomarkers (Apoe4, ADAS11, MPACC digits, and Intracranial Volume).

3. **Multimodal Integration**:
   - The study demonstrates the efficacy of combining neuroimaging data with complementary non-imaging biomarkers, achieving state-of-the-art (SOTA) accuracy in multiclass classification tasks.

## Results and Impact

The proposed framework achieves SOTA performance with a mean accuracy of **84.87±6.09%** and an F1 score of **84.83±6.12%** for CN vs. MCI vs. AD classification. Furthermore:
- It generalizes well to binary classification tasks, achieving perfect accuracy (100%) in distinguishing between AD and CN cases.
- It significantly improves interpretability and stability compared to existing sequential binary classification methods.

### Comparison to Existing Methods
Unlike traditional methods, which often decompose multiclass tasks into sequential binary classifications, this framework directly tackles multiclass classification, enabling more comprehensive insights into the progression of Alzheimer's disease. The regularized multikernel SVM prevents overfitting and ensures a balanced representation of all modalities, offering a scalable solution for similar neuroimaging challenges.

### Insights into Neuroimaging and Biomarkers
The study highlights the relative contributions of different brain regions and modalities to classification accuracy, offering valuable insights into the biological underpinnings of Alzheimer's disease. For example:
- The left hippocampus and left amygdala were identified as critical regions for early detection.
- The integration of cognitive scores (e.g., ADAS11) significantly enhanced predictive power.

## Conclusion

This research presents a significant advancement in the field of Alzheimer's disease classification by introducing a unified, interpretable, and high-performing multiclass framework. The integration of advanced embedding techniques with regularized multikernel learning provides a pathway for robust multimodal analyses, with potential applications extending beyond Alzheimer's disease to other neurodegenerative disorders.

## Citation
If you use this code or framework, please cite:
```
Oyekanmi O. Olatunde, Kehinde S. Oyetunde, Jihun Han, Mohammad T. Khasawneh, Hyunsoo Yoon. 
"Multiclass classification of Alzheimer's disease prodromal stages using sequential feature embeddings and regularized multikernel support vector machine". NeuroImage, 2024.
DOI: https://doi.org/10.1016/j.neuroimage.2024.120929
```
