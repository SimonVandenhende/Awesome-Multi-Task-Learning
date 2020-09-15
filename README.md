# Awesome Multi-Task Learning
This page contains a list of papers on multi-task learning in the deep learning era. 
Please create a pull request if you wish to add anything. 
If you are interested, consider reading our recent survey paper (an update will follow soon).

```
@article{vandenhende2020revisiting,
  title={Revisiting Multi-Task Learning in the Deep Learning Era},
  author={Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Dai, Dengxin and Van Gool, Luc},
  journal={arXiv preprint arXiv:2004.13379},
  year={2020}
}
```

## Table of Contents:

- [Survey papers](#survey) 
- [Datasets](#datasets)
- [Architectures](#architectures)
  - [Encoder-based](#encoder)
  - [Decoder-based](#decoder)
  - [Other](#otherarchitectures)
- [Neural Architecture Search](#nas)
- [Optimization strategies](#optimization)
- [Transfer learning](#transfer)


<a name="survey"></a>
## Survey papers
- <a name="vandenhende2020revisiting"></a> Vandenhende, S., Georgoulis, S., Proesmans, M., Dai, D., & Van Gool, L. 
*[Revisiting Multi-Task Learning in the Deep Learning Era](https://arxiv.org/abs/2004.13379)*,
ArXiv, 2020. [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]

- <a name="ruder2017survey"></a> Ruder, S. 
*[An overview of multi-task learning in deep neural networks](https://arxiv.org/abs/1706.05098)*,
ArXiv, 2017. 

- <a name="zhang2017survey"></a> Zhang, Y.
*[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)*, 
ArXiv, 2017.

- <a name="gong2019comparison"></a> Gong, T., Lee, T., Stephenson, C., Renduchintala, V., Padhy, S., Ndirango, A., ... & Elibol, O. H. 
*[A comparison of loss weighting strategies for multi task learning in deep neural networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8848395)*,
IEEE Access, 2019. 


<a name="datasets"></a>
## Datasets
The following datasets have been regularly used in the context of multi-task learning:

- [NYUDv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [PASCAL](https://github.com/facebookresearch/astmt)
- [Taskonomy](https://github.com/StanfordVL/taskonomy)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [SUN RGB-D](https://rgbd.cs.princeton.edu/)
- [BDD100K](https://arxiv.org/pdf/1805.04687.pdf)

<a name="architectures"></a>
## Architectures

<a name="encoder"></a>
### Encoder-based architectures

- <a name="misra2016cross"></a> Misra, I., Shrivastava, A., Gupta, A., & Hebert, M.
*[Cross-stitch networks for multi-task learning](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html)*,
CVPR, 2016. [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]

- <a name="gao2019nddr"></a> Gao, Y., Ma, J., Zhao, M., Liu, W., & Yuille, A. L. 
*[Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_NDDR-CNN_Layerwise_Feature_Fusing_in_Multi-Task_CNNs_by_Neural_Discriminative_CVPR_2019_paper.html)*,
CVPR, 2019. [[Tensorflow](https://github.com/ethanygao/NDDR-CNN)] [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]

- <a name="liu2019mtan"></a> Liu, S., Johns, E., & Davison, A. J. 
*[End-to-end multi-task learning with attention](https://arxiv.org/abs/1803.10704)*,
CVPR, 2019. [[PyTorch](https://github.com/lorenmt/mtan)]


<a name="decoder"></a>
### Decoder-based architectures

- <a name="xu2018pad"></a> Xu, D., Ouyang, W., Wang, X., & Sebe, N.
*[Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html)*,
CVPR, 2018.  

- <a name="zhang2018jtrl"></a> Zhang, Z., Cui, Z., Xu, C., Jie, Z., Li, X., & Yang, J.
*[Joint task-recursive learning for semantic segmentation and depth estimation](https://openaccess.thecvf.com/content_ECCV_2018/html/Zhenyu_Zhang_Joint_Task-Recursive_Learning_ECCV_2018_paper.html)*,
ECCV, 2018.

- <a name="ruder2019sluice"></a> Ruder, S., Bingel, J., Augenstein, I., & Søgaard, A. 
*[Latent multi-task architecture learning](https://www.aaai.org/ojs/index.php/AAAI/article/view/4410)*,
AAAI, 2019.

- <a name="zhang2019papnet"></a> Zhang, Z., Cui, Z., Xu, C., Yan, Y., Sebe, N., & Yang, J. 
*[Pattern-affinitive propagation across depth, surface normal and semantic segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.html)*,
CVPR, 2019.

- <a name="zhou2020structure"></a> Zhou, L., Cui, Z., Xu, C., Zhang, Z., Wang, C., Zhang, T., & Yang, J.
*[Pattern-Structure Diffusion for Multi-Task Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Pattern-Structure_Diffusion_for_Multi-Task_Learning_CVPR_2020_paper.html)*,
CVPR, 2020.

- <a name="vandenhende2020mti"></a> Vandenhende, S., Georgoulis, S., & Van Gool, L. 
*[MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning](https://arxiv.org/abs/2001.06902)*,
ECCV, 2020. [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]
  
<a name="otherarchitectures"></a>

### Other

- <a name="yang2016deep"></a> Yang, Y., & Hospedales, T. 
*[Deep multi-task representation learning: A tensor factorisation approach](https://arxiv.org/abs/1605.06391)*,
ICLR, 2017.

- <a name="kokkinos2017uber"></a> Kokkinos, Iasonas.
*[Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory](https://openaccess.thecvf.com/content_cvpr_2017/html/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.html)*,
CVPR, 2017.

- <a name="rebuffi2017learning"></a> Rebuffi, S. A., Bilen, H., & Vedaldi, A. 
*[Learning multiple visual domains with residual adapters](https://arxiv.org/abs/1705.08045)*,
NIPS, 2017.

- <a name="long2017multilinear"></a> Long, M., Cao, Z., Wang, J., & Philip, S. Y. 
*[Learning multiple tasks with multilinear relationship networks](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-deep-relationship-networks)*,
NIPS, 2017.

- <a name="meyerson2017beyond"></a> Meyerson, E., & Miikkulainen, R. 
*[Beyond shared hierarchies: Deep multitask learning through soft layer ordering](https://arxiv.org/abs/1711.00108)*,
ICLR, 2018.

- <a name="rosenbaum2017routing"></a> Rosenbaum, C., Klinger, T., & Riemer, M.
*[Routing networks: Adaptive selection of non-linear functions for multi-task learning](https://arxiv.org/abs/1711.01239)*,
ICLR, 2018.

- <a name="mallya2018piggy"></a> Mallya, A., Davis, D., & Lazebnik, S.
*[Piggyback: Adapting a single network to multiple tasks by learning to mask weights](https://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html)*,
ECCV, 2018.

- <a name="rebuffi2018efficient"></a> Rebuffi, S. A., Bilen, H., & Vedaldi, A.
*[Efficient parametrization of multi-domain deep neural networks](https://arxiv.org/abs/1803.10082)*,
CVPR, 2018.

- <a name="maninis2019astmt"></a> Maninis, K. K., Radosavovic, I., & Kokkinos, I. 
*[Attentive single-tasking of multiple tasks](https://arxiv.org/abs/1904.08918)*,
CVPR, 2019. [[PyTorch](https://github.com/facebookresearch/astmt)]

- <a name="kanakis2020reparameterizing"></a> Kanakis, M., Bruggemann, D., Saha, S., Georgoulis, S., Obukhov, A., & Van Gool, L.
*[Reparameterizing Convolutions for Incremental Multi-Task Learning without Task Interference](https://arxiv.org/abs/2007.12540)*,
ECCV, 2020.


<a name="nas"></a>
## Neural Architecture Search

- <a name="lu2017fully"></a> Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., & Feris, R.
*[Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification](https://openaccess.thecvf.com/content_cvpr_2017/html/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.html)*,
CVPR, 2017. 

- <a name="sun2019adashare"></a> Sun, X., Panda, R., & Feris, R. 
*[AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning](https://arxiv.org/abs/1911.12423)*,
ArXiv, 2019.

- <a name="bragman2019stochastic"></a> Bragman, F. J., Tanno, R., Ourselin, S., Alexander, D. C., & Cardoso, J.
*[Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels](https://openaccess.thecvf.com/content_ICCV_2019/html/Bragman_Stochastic_Filter_Groups_for_Multi-Task_CNNs_Learning_Specialist_and_Generalist_ICCV_2019_paper.html)*,
ICCV, 2019.

- <a name="newell2019feature"></a> Newell, A., Jiang, L., Wang, C., Li, L. J., & Deng, J. 
*[Feature partitioning for efficient multi-task architectures](https://arxiv.org/abs/1908.04339)*,
 ArXiv, 2019.

- <a name="guo2020learning"></a> Guo, P., Lee, C. Y., & Ulbricht, D. 
*[Learning to Branch for Multi-Task Learning](https://proceedings.icml.cc/static/paper_files/icml/2020/5057-Paper.pdf)*, 
ICML, 2020. 

- <a name="standley2019tasks"></a> Standley, T., Zamir, A. R., Chen, D., Guibas, L., Malik, J., & Savarese, S. 
*[Which Tasks Should Be Learned Together in Multi-task Learning?](https://arxiv.org/pdf/1905.07553.pdf)*,
ICML, 2020.

- <a name="vandenhende2019branched"></a> Vandenhende, S., Georgoulis, S., De Brabandere, B., & Van Gool, L. 
*[Branched multi-task networks: deciding what layers to share](https://arxiv.org/abs/1904.02920)*, 
BMVC, 2020. 

- <a name="bruggeman2020auomated"></a> Bruggemann, D., Kanakis, M., Georgoulis, S., & Van Gool, L.
*[Automated Search for Resource-Efficient Branched Multi-Task Networks](https://arxiv.org/abs/2008.10292)*,
BMVC, 2020.

<a name="optimization"></a>
## Optimization strategies

- <a name="kendall2018uncertainty"></a> Kendall, A., Gal, Y., & Cipolla, R. 
*[Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)*,
CVPR, 2018. 

- <a name="zhao2018modulation"></a> Zhao, X., Li, H., Shen, X., Liang, X., & Wu, Y. 
*[A modulation module for multi-task learning with applications in image retrieval](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiangyun_Zhao_A_Modulation_Module_ECCV_2018_paper.html)*,
ECCV, 2018.

- <a name="chen2018gradnorm"></a> Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. 
*[Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks](http://proceedings.mlr.press/v80/chen18a.html)*,
ICML, 2018.

- <a name="sener2018mgda"></a> Sener, O., & Koltun, V. 
*[Multi-task learning as multi-objective optimization](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization)*,
NIPS, 2018. [[PyTorch](https://github.com/intel-isl/MultiObjectiveOptimization)]

- <a name="liu2017adversarial"></a> Liu, P., Qiu, X., & Huang, X.
*[Adversarial multi-task learning for text classification](https://www.aclweb.org/anthology/P17-1001.pdf)*,
ACL, 2018.

- <a name="guo2018dynamic"></a> Guo, M., Haque, A., Huang, D. A., Yeung, S., & Fei-Fei, L.
*[Dynamic task prioritization for multitask learning](https://openaccess.thecvf.com/content_ECCV_2018/html/Michelle_Guo_Focus_on_the_ECCV_2018_paper.html)*,
ECCV, 2018. 

- <a name="lin2019pareto"></a> Lin, X., Zhen, H. L., Li, Z., Zhang, Q. F., & Kwong, S.
*[Pareto multi-task learning](https://papers.nips.cc/paper/9374-pareto-multi-task-learning)*,
NIPS, 2019.

- <a name="suteu2019orthogonal"></a> Suteu, M., & Guo, Y. 
*[Regularizing Deep Multi-Task Networks using Orthogonal Gradients](https://arxiv.org/abs/1912.06844)*,
ArXiv, 2019. 

- <a name="yu2020surgery"></a> Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. 
*[Gradient surgery for multi-task learning](https://arxiv.org/abs/2001.06782)*,
ArXiv, 2020. [[Tensorflow](https://github.com/tianheyu927/PCGrad)]


<a name="transfer"></a>
## Transfer learning

- <a name="cui2018large"></a> Cui, Y., Song, Y., Sun, C., Howard, A., & Belongie, S.
*[Large scale fine-grained categorization and domain-specific transfer learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.html)*,
CVPR, 2018.

- <a name="zamir2018taskonomy"></a> Zamir, A. R., Sax, A., Shen, W., Guibas, L. J., Malik, J., & Savarese, S.
*[Taskonomy: Disentangling task transfer learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.html)*,
CVPR, 2018. [[PyTorch](https://github.com/StanfordVL/taskonomy)]

- <a name="achille2019task2vec"></a> Achille, A., Lam, M., Tewari, R., Ravichandran, A., Maji, S., Fowlkes, C. C., ... & Perona, P.
*[Task2vec: Task embedding for meta-learning](https://openaccess.thecvf.com/content_ICCV_2019/html/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html)*,
ICCV, 2019. [[PyTorch](https://github.com/awslabs/aws-cv-task2vec)]

- <a name="dwivedi2019rsa"></a> Dwivedi, K., & Roig, G.
*[Representation similarity analysis for efficient task taxonomy & transfer learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwivedi_Representation_Similarity_Analysis_for_Efficient_Task_Taxonomy__Transfer_Learning_CVPR_2019_paper.html)*,
CVPR, 2019. [[PyTorch](https://github.com/kshitijd20/RSA-CVPR19-release)]

<a name="robustness"></a>
## Robustness
- <a name="maomultitask2020"></a> Mao, C., Gupta, A., Nitin, V., Ray, B., Song, S., Yang, J., & Vondrick, C.
*[Multitask Learning Strengthens Adversarial Robustness](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470154.pdf)*,
ECCV, 2020. [[PyTorch](https://github.com/columbia/MTRobust)]

- <a name="zamirrobust2020"></a> Zamir, A. R., Sax, A., Cheerla, N., Suri, R., Cao, Z., Malik, J., & Guibas, L. J. 
*[Robust Learning Through Cross-Task Consistency](https://openaccess.thecvf.com/content_CVPR_2020/html/Zamir_Robust_Learning_Through_Cross-Task_Consistency_CVPR_2020_paper.html)*,
CVPR, 2020. 


