# Awesome Out-of-distribution detection(OOD) papers with taxonomy tag

## Preliminary
This repository is a collection of significant works published at top-tier conference in the topic of out-of-distribution detection. To facilitate better paper reading, papers are listed with their code (if publicly available) along with their tags in terms of (settings, modalities, approaches, etc.). Below are explanation for corresponding tags that might be useful for readers.

## Tags for OOD papers
![Posthoc]: An OOD scoring function to detect OOD samples is employed based on the pretrained deep neural network (could be logit-based or feature-based).

![Training]: Training procedure is modified to faciliate model's capability in detecting out-of-distribution samples 

![Theoretical Analysis/Insights]: Theoretical justification for method is provided, or theoretical analysis is investigated to study the behavior of model on OOD data.

![Adversarial]

![Benchmark/Observational Analysis]: New benchmark or problem setting is provided. Experimental results are thoroughly observed to provide some insights, novel findings, etc.

![InputPreprocessing]: Inputs are preprocessed(pertubated) before being fed to the model.


## List of OOD papers

### NeurIPS

![](https://img.shields.io/badge/NeurIPS2017-blue) &nbsp; [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)

![](https://img.shields.io/badge/NeurIPS2018-blue) &nbsp; [Out-of-Distribution Detection using Multiple Semantic Label Representations](https://proceedings.neurips.cc/paper_files/paper/2018/file/2151b4c76b4dcb048d06a5c32942b6f6-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2018-blue) &nbsp; [Predictive Uncertainty Estimation via Prior Networks](https://arxiv.org/pdf/1802.10501.pdf)

![](https://img.shields.io/badge/NeurIPS2018-blue) &nbsp; [A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/pdf/1807.03888.pdf)

![](https://img.shields.io/badge/NeurIPS2019-blue) &nbsp; [Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices](https://arxiv.org/pdf/1912.12510.pdf)

![](https://img.shields.io/badge/NeurIPS2019-blue) &nbsp; [Likelihood Ratios for Out-of-Distribution Detection](https://proceedings.neurips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf)      

![](https://img.shields.io/badge/NeurIPS2019-blue) &nbsp; [Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty](https://proceedings.neurips.cc/paper/2019/file/a2b15837edac15df90721968986f7f8e-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2019-blue) &nbsp; [Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift.](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2020-blue) &nbsp; [Energy-based Out-of-distribution Detection](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2020-blue) &nbsp; [Csi: Novelty detection via contrastive learning on distributionally shifted instances](https://proceedings.neurips.cc/paper/2020/file/8965f76632d7672e7d3cf29c87ecaa0c-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2020-blue) &nbsp; [Measuring Robustness to Natural Distribution Shifts in Image Classification](https://proceedings.neurips.cc/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2021-blue) &nbsp; [Can multi-label classification networks know what they don’t know?](https://proceedings.neurips.cc/paper/2021/file/f3b7e5d3eb074cde5b76e26bc0fb5776-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2021-blue) &nbsp; [ReAct: Out-of-distribution Detection With Rectified Activations](https://arxiv.org/pdf/2111.12797.pdf)

![](https://img.shields.io/badge/NeurIPS2021-blue) &nbsp; [On the Importance of Gradients for Detecting Distributional Shifts in the Wild](https://proceedings.neurips.cc/paper/2021/file/063e26c670d07bb7c4d30e6fc69fe056-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2021-blue) &nbsp; [Exploring the Limits of Out-of-Distribution Detection](https://proceedings.neurips.cc/paper/2021/file/3941c4358616274ac2436eacf67fae05-Paper.pdf)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Density-driven Regularization for Out-of-distribution Detection](https://openreview.net/pdf?id=aZQJMVx8fk)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [UQGAN: A Unified Model for Uncertainty Quantification of Deep Classifiers trained via Conditional GANs](https://openreview.net/pdf?id=djOANbV2zSu)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://openreview.net/pdf?id=KnCS9390Va)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [SIREN: Shaping Representations for Detecting Out-of-Distribution Objects](https://openreview.net/pdf?id=8E8tgnYlmN)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Is Out-of-Distribution Detection Learnable?](https://openreview.net/pdf?id=sde_7ZzGXOE)


![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Your Out-of-Distribution Detection Method is Not Robust!](https://openreview.net/pdf?id=YUEP3ZmkL1)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection](https://openreview.net/pdf?id=-deKNiSOXLG)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Out-of-Distribution Detection with An Adaptive Likelihood Ratio on Informative Hierarchical VAE](https://openreview.net/pdf?id=vMQ1V_z0TxU)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Boosting Out-of-distribution Detection with Typical Features](https://openreview.net/pdf?id=4maAiUt0A4)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Beyond Mahalanobis Distance for Textual OOD Detection](https://openreview.net/pdf?id=ReB7CCByD6U)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Out-of-Distribution Detection via Conditional Kernel Independence Model](https://openreview.net/pdf?id=rTTh1RIn6E)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [GraphDE: A Generative Framework for Debiased Learning and Out-of-Distribution Detection on Graphs](https://openreview.net/pdf?id=mSiPuHIP7t8)

![](https://img.shields.io/badge/NeurIPS2022-blue) &nbsp; [Deep Ensembles Work, But Are They Necessary?](https://arxiv.org/pdf/2202.06985.pdf)

![](https://img.shields.io/badge/NeurIPS2022-blue)

![](https://img.shields.io/badge/NeurIPS2022-blue)

### ICLR

![](https://img.shields.io/badge/ICLR2017-violet) &nbsp; [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
](https://arxiv.org/pdf/1610.02136.pdf)

![](https://img.shields.io/badge/ICLR2018-violet) &nbsp; [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples
](https://arxiv.org/pdf/1711.09325.pdf)

![](https://img.shields.io/badge/ICLR2018-violet) &nbsp; [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks
](https://arxiv.org/pdf/1706.02690.pdf)

![](https://img.shields.io/badge/ICLR2019-violet) &nbsp; [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/pdf/1810.09136.pdf)

![](https://img.shields.io/badge/ICLR2019-violet) &nbsp; [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/pdf/1812.04606.pdf)

![](https://img.shields.io/badge/ICLR2020-violet) &nbsp; [Input complexity and out-of-distribution detection with likelihood-based generative models
](https://arxiv.org/pdf/1909.11480.pdf)

![](https://img.shields.io/badge/ICLR2021-violet) &nbsp; [In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness
](https://arxiv.org/pdf/2012.04550.pdf)

![](https://img.shields.io/badge/ICLR2021-violet) &nbsp; [Removing Undesirable Feature Contributions Using Out-of-Distribution Data](https://arxiv.org/pdf/2101.06639.pdf)

![](https://img.shields.io/badge/ICLR2021-violet) &nbsp; [Multiscale Score Matching for Out-of-Distribution Detection](https://arxiv.org/pdf/2010.13132.pdf)

![](https://img.shields.io/badge/ICLR2021-violet) &nbsp; [Protecting DNNs from Theft using an Ensemble of Diverse Models](https://openreview.net/pdf?id=LucJxySuJcE)

![](https://img.shields.io/badge/ICLR2021-violet) &nbsp; [SSD: A Unified Framework for Self-Supervised Outlier Detection](https://openreview.net/pdf?id=v5gjXpmR8J)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/pdf/2202.10054.pdf)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [PI3NN: Out-of-distribution-aware prediction intervals from three neural networks](https://arxiv.org/pdf/2108.02327.pdf)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [Meta Learning Low Rank Covariance Factors for Energy Based Deterministic Uncertainty](https://openreview.net/pdf?id=GQd7mXSPua)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://openreview.net/pdf?id=TW7d65uYu5M)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [A Statistical Framework for Efficient Out of Distribution Detection in Deep Neural Networks](https://openreview.net/pdf?id=Oy9WeuZD51)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [Revisiting flow generative models for Out-of-distribution detection](https://openreview.net/pdf?id=6y2KBh-0Fd9)

![](https://img.shields.io/badge/ICLR2022-violet) &nbsp; [Igeood: An Information Geometry Approach to Out-of-Distribution Detection](https://openreview.net/pdf?id=mfwdY3U_9ea)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Extremely Simple Activation Shaping for Out-of-Distribution Detection](https://openreview.net/forum?id=ndYXTEL6cZz)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Non-parametric Outlier Synthesis](https://openreview.net/forum?id=JHklpEZqduQ)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Efficient Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy](https://openreview.net/forum?id=KkazG4lgKL)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?](https://openreview.net/forum?id=aEFaE0W5pAd)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Out-of-distribution Detection with Implicit Outlier Transformation](https://openreview.net/forum?id=hdghx6wbGuD)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Energy-based Out-of-Distribution Detection for Graph Neural Networks](https://openreview.net/forum?id=zoz7Ze4STUL)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Harnessing Out-Of-Distribution Examples via Augmenting Content and Style](https://openreview.net/forum?id=boNyg20-JDm)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection](https://openreview.net/forum?id=mMNimwRb7Gr)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [A framework for benchmarking Class-out-of-distribution detection and its application to ImageNet](https://openreview.net/forum?id=Iuubb9W6Jtk)

![](https://img.shields.io/badge/ICLR2023-violet) &nbsp; [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://openreview.net/forum?id=kJUS5nD0vPB)


### CVPR

![](https://img.shields.io/badge/CVPR19-green) &nbsp; [Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hein_Why_ReLU_Networks_Yield_HighConfidence_Predictions_Far_Away_From_the_CVPR_2019_paper.pdf)

![](https://img.shields.io/badge/CVPR20-green) &nbsp; [Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.pdf)

![](https://img.shields.io/badge/CVPR20-green) &nbsp; [Deep Residual Flow for Out of Distribution Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zisselman_Deep_Residual_Flow_for_Out_of_Distribution_Detection_CVPR_2020_paper.pdf)

![](https://img.shields.io/badge/CVPR21-green) &nbsp; [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_MOS_Towards_Scaling_Out-of-Distribution_Detection_for_Large_Semantic_Space_CVPR_2021_paper.pdf)

![](https://img.shields.io/badge/CVPR21-green) &nbsp; [Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces](https://openaccess.thecvf.com/content/CVPR2021/papers/Zaeemzadeh_Out-of-Distribution_Detection_Using_Union_of_1-Dimensional_Subspaces_CVPR_2021_paper.pdf)

![](https://img.shields.io/badge/CVPR21-green) &nbsp; [MOOD: Multi-level Out-of-distribution Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_MOOD_Multi-Level_Out-of-Distribution_Detection_CVPR_2021_paper.pdf)

![](https://img.shields.io/badge/CVPR22-green) &nbsp; [Deep Hybrid Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Deep_Hybrid_Models_for_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)

![](https://img.shields.io/badge/CVPR22-green) &nbsp; [Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection
](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Reconstruction_Autoencoder-Based_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)

![](https://img.shields.io/badge/CVPR22-green) &nbsp; [ViM: Out-Of-Distribution with Virtual-logit Matching](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf)

![](https://img.shields.io/badge/CVPR22-green) &nbsp; [Weakly Supervised Semantic Segmentation using Out-of-Distribution Data](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Weakly_Supervised_Semantic_Segmentation_Using_Out-of-Distribution_Data_CVPR_2022_paper.pdf)

![](https://img.shields.io/badge/CVPR22-green) &nbsp; [Neural Mean Discrepancy for Efficient Out-of-Distribution Detection
](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Neural_Mean_Discrepancy_for_Efficient_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)

![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Rethinking Out-of-distribution (OOD) Detection:
Masked Image Modeling is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf)

![](https://img.shields.io/badge/CVPR23-green) &nbsp; [LINe: Out-of-Distribution Detection by Leveraging Important Neurons
](https://openaccess.thecvf.com/content/CVPR2023/papers/Ahn_LINe_Out-of-Distribution_Detection_by_Leveraging_Important_Neurons_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Balanced Energy Regularization Loss for Out-of-Distribution Detection
](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Decoupling MaxLogit for Out-of-Distribution Detection
](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Detection of out-of-distribution samples using binary neuron activation patterns](https://openaccess.thecvf.com/content/CVPR2023/papers/Olber_Detection_of_Out-of-Distribution_Samples_Using_Binary_Neuron_Activation_Patterns_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Uncertainty-Aware Optimal Transport for Semantically Coherent
Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Uncertainty-Aware_Optimal_Transport_for_Semantically_Coherent_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)


![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Distribution Shift Inversion for Out-of-Distribution Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf)

![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Block Selection Method for Using Feature Norm in Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Block_Selection_Method_for_Using_Feature_Norm_in_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)

![](https://img.shields.io/badge/CVPR23-green) &nbsp; [Are Data-driven Explanations Robust against Out-of-distribution Data?](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Are_Data-Driven_Explanations_Robust_Against_Out-of-Distribution_Data_CVPR_2023_paper.pdf)

### ICML


### Other venues

[InputPreprocessing]: https://img.shields.io/badge/InputPreprocessing-f4d5b3?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAABDUlEQVQ4jc3TPy8EURQF8N8uS/wJGxuh0tH7CBKthk/gk6iIQiFRSEhEFEQhGoSQbERUEo2SGp1CwTa7infJZE3sbuckr5j75p5z7pk7/BesoIZGm6eG5SxBDSMdCFbwmS002mi6xU1zT7ED1fpfQtmLAexhtAVhI++hGyd4wD3KUS/jUJr9G8P8HmETBUzjGqdSuMeYwno4PMMjZrMOlnCHwagVsI23UC9iHNWoz+AlS/CEsSZHXTjABvpwiZ0YdSsc/hBMykcJEziXwi0FSTXGQVqkSl43ekNpHz1BcoV+YQXW8BwvZLGKVymPRexKoc7hQ1y0whHepXzqWJBZ41abWJA+3xAuMK/pH/gCPJhBnIabIDQAAAAASUVORK5CYII=

[Training]: https://img.shields.io/badge/Training-f4d5b3?style=social&logo=data:image/png;base64,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

[Posthoc]: https://img.shields.io/badge/Posthoc-f4d5b3?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAYAAADimHc4AAAABmJLR0QA/wD/AP+gvaeTAAAFXElEQVR4nO2dW2gdRRjHf7bRIiKiVNSK2IqkCV4ajdfUqEWrSOKt1KpVUi+NN0SliD5oVfClCj4Vi/oo6oNI38UHxUbEB29QwUtVvEWbqmlMTc31+DCNrbDz7ew5szu7O98Pljxk58+3//85s7O7s3NAURRFURRFURRFURRFUVpnAbAGeAP4FpgAGgG3F4HDcj3iEnEh8BlhDY82hJuA/YQ3O8oQVgL/EN7ktO0VTBdZK44AdhHe3GhDuJ/wpkbdHb1H8kH+AQwAxxZYy3JLLbUNoQ2YJvkA+wPUkyWAWoRwMskHtpcwB5Y1gMqfE2wH/GXJ6ildCJVNvEmGgc+F/w8CWynwWxtbAOPAKuBjYZ8HgJcpyJvYAgAYBVYjhzAIvEQB/sQYAJQohFgDgJKEEHMAUIIQYg8AAoegARiChdDmU6wCzF+gNcvggb/3AXOtl6PfgGYYBF7wJaYBNMc1voQ0gMBoAIHRAAJT1wC+wtzRbHXryLvQugZQGTSAwGgAgdEAAqMBBCa2e0FZmR9N5YZ+AwKjAQRGAwiMBhAYDSAwGoAfjgc2Ax8Bu4ExYCdmlt1ZRRRQtrmhRXInxnDbfNNZYBvmBZbciDWAZ3Cf+PsucFRehcQYwNNkn309BBydRzGxBfAozU1/zy2EmAJ4jObNP7Q78npOiCWATcjG7gJWAIuB7Sn7bvNZWAwBPISZjGUz9Adg6SH7LwReF/afBbp8FVf3ADYim/8jsCyhXVoIW30VWOcABkk3/zShfRv27minryLrGsBdmK7CZv7PwOkOOost7cd8FVrHAAaQzR/GHLcLZ1g09vgqtm4BrAdmsJv/G9DpqHUi8IVF50NfBdcpgHXI5o8AZzpqLcF4YNN6ylfRdQlgDTCF3bBRoNtR6xTga0FrHDjBV+F1COB6ZPP/BM5x1DoVs0ybTavBwZc9vFD1APqBSeRP/nmOWsuA7wWtBvCcx9qBagdwFfISa2OYNfBcWAp8J2g1gOc91v4fVQ1gNbL5+4BLHbXagZ8ErQawxWPt/6OKAfRiDLaZ9TdwuaPWcuAXQStX8+cLqFIAKzGjEMn8VY5aHZiLMsn8zR5rT6RKAfQAf2E3awK4wlGrE/hV0GoAT3is3UpVArgI2fxJoM9RqwtzK8GmNQc84rF2kSoEcC5mLC+Zf20Grd8FrTnM84PCKHsAXZjVG22GTQHXOWp1p2jNAQ96rN2JMgewAvnTOgPc7KjVgzwHaA6z4lbhlDWAs5H76RngVketS5DPH7PAHR5rz8QSS1FjhFuPswN5hDID3Oao1Yts/gywwWPtmSnbwq2dmHv2kmG3O2pdibkusGlNA7d4rL1ppKWLNwDHFVRHO/KFUZau4mrkH56YAtZ6rL0l7sVeqI/NhWMwz2mlk+Tdjlp9yEvxTwI3OmoVwuHAN4QNoE9on2WE0od8k24SuMFRq1B6yO8HHFzYYmmbZWzen3IMk7hfMwRhLfn8hIkLQ5a2zzq2X4f8VGwC8/yg9JwPfEKxARyJ/ZPrMoNhPfaRXAMzEnK9SVcKFmD6ydcw54ZWf8Yqjcss7UZIvxYZQJ4JMX5AXxF4kmTztqe0S5v9tg/3ZwNR8zbJBm4S2mxENn8vcHF+JdeHhdhvkl2QsH8HZlKUNOl21NJWSaAbe/exCDM38x7gVcxs5rTzjZqfkYdJNnI/8k20pG0PHl+ciIW38DPUHcHcwlYykjYrwWXbTUFvtdeNdlo3f5gClqhMog4rZvU20WYas1T9ELADeB8z5CycWAKYAD7FGP4BxnBvrwfFTtI08DHgHeBxzLPcRcGqqzkncbAPfxMzHO2mQsvw/Atv0E+fkVDOBwAAAABJRU5ErkJggg==

[Theoretical Analysis/Insights]: https://img.shields.io/badge/Theoretical_Analysis/Insights-f4d5b3?style=social&logo=data:image/png;base64,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

[Benchmark/Observation]: https://img.shields.io/badge/Benchmark/Observational__Analysis-f4d5b3?style=social&logo=bookmeter

[Adversarial]:https://img.shields.io/badge/Adversarial-f4d5b3?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAtCAAAAAAfcb1GAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+cGEwMhOKyin28AAAJtSURBVEjHY/hPCmAYVqp/f/1NtOpPcwMcAuZ+Jk7143DDpuVNhhGPiVH92j/g5rfv3y64+r0mrPp3vturs76enj4WXIW/Caper3P2/7ujhw4dOrpGZz0h1W/su4HktbTEpKQEeZu3BFTPsnoFJF+sWb1mzdqV5rPwq/7oOBXCOFRaVlZuaPsRr+odxk8gjLP9EydO6NDegVd1QfyDE+tndLc0tvXP2XzmcXwBPtWfbaUUhdOa6urqauua0oQVpWw/41H9WJGBgX3f//9//wA5+9gZGBQf41H9yZWBgWHG///35gOVzwCyXT/hc/cxZQYGr6//Hxe9/v/Vi4FB+RheX76MZWBg6/z9/9f/351sDAyxL/Go/r3KlAVoPVfKjos7UriALBbTVb9xqf7RwMEAASz8LFAWR8MPHKrXcBtFREaAQTiEigzX5VyDXfXPAI8LO7aDwLYtm7duAzN2nbMI+IlV9VmhnilsHEDAzqXrrMnBDmJxLC4VPItVdT3DhA6wW7kaH3+5V8oOZk+rYqjHpvqdGUx19l+QlyPB7OlVDGbvsKg+ygdVzbETzF/JDFXNdxSL6pkMUNXcK0r7L7ZXL2SDqmaYiUV1LUw1Q0Vx+vHYmmyYSxhqsahuhavO+L80f8O/cLjqViyqj4vCVLtNP33/+ARrmGrR41hU/+tj64OoZuBbdmEGNOqnl7P2/cMW3r+nLGuHpg4uHk4oa9qkyUjpCjmd/PvVIqSiDAJKSkpgWkVwyrd//7Gr/v+/uejsKWRwNm0Sijyq6iZBVVTAPxmP6g221jbIwNphLx7Vf799RQUorh78dRp1VAMA3gZbAjW7k9kAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjMtMDYtMTlUMDM6MzM6NDkrMDA6MDAsKP/ZAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIzLTA2LTE5VDAzOjMzOjQ5KzAwOjAwXXVHZQAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyMy0wNi0xOVQwMzozMzo1NiswMDowMDCCFs0AAAAASUVORK5CYII=

