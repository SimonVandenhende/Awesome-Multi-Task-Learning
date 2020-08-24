# Awesome Multi Task Learning
This page contains a list of papers and projects on multi-task learning. 

## Table of Contents:

- [Survey paper](#survey) 
- [Datasets](#datasets)
- [Architectures](#architectures)
  - [Encoder-based](#encoder)
  - [Decoder-based](#decoder)
  - [Other](#otherarchitectures)
- [Neural Architecture Search](#nas)
- [Optimization strategies](#optimization)


<a name="survey"></a>
## Survey paper
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

- <a name="zhang2019papnet"></a> Zhang, Z., Cui, Z., Xu, C., Yan, Y., Sebe, N., & Yang, J. 
*[Pattern-affinitive propagation across depth, surface normal and semantic segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.html)*,
CVPR, 2019.

- <a name="vandenhende2020mti"> Vandenhende, S., Georgoulis, S., & Van Gool, L. 
*[MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning](https://arxiv.org/abs/2001.06902)*,
ECCV, 2020. [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]
  
<a name="otherarchitectures"></a>
### Other

- <a name="maninis2019astmt"></a> Maninis, K. K., Radosavovic, I., & Kokkinos, I. 
*[Attentive single-tasking of multiple tasks.](https://arxiv.org/abs/1904.08918)*,
CVPR, 2019. [[PyTorch](https://github.com/facebookresearch/astmt)]

<a name="nas"></a>
## Neural Architecture Search

- <a name="lu2017fully"></a> Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., & Feris, R.
*[Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification.](https://openaccess.thecvf.com/content_cvpr_2017/html/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.html)*,
CVPR, 2017. 

- <a name="sun2019adashare"></a> Sun, X., Panda, R., & Feris, R. 
*[AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning.](https://arxiv.org/abs/1911.12423)*,
ArXiv, 2019.

- <a name="vandenhende2019branched"></a> Vandenhende, S., Georgoulis, S., De Brabandere, B., & Van Gool, L. 
*[Branched multi-task networks: deciding what layers to share.](https://arxiv.org/abs/1904.02920)*, 
BMVC, 2020. 

- <a name="guo2020learning"></a> Guo, P., Lee, C. Y., & Ulbricht, D. 
*[Learning to Branch for Multi-Task Learning.](https://proceedings.icml.cc/static/paper_files/icml/2020/5057-Paper.pdf)*, 
ICML, 2020. 


<a name="optimization"></a>
## Optimization strategies

- <a name="kendall2018uncertainty"></a> Kendall, A., Gal, Y., & Cipolla, R. 
*[Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)*,
CVPR, 2018. 

- <a name="chen2018gradnorm"></a> Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. 
*[Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks.](http://proceedings.mlr.press/v80/chen18a.html)*,
ICML, 2018.

- <a name="sener2018mgda"></a> Sener, O., & Koltun, V. 
*[Multi-task learning as multi-objective optimization.](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization)*,
NIPS, 2018. [[PyTorch](https://github.com/intel-isl/MultiObjectiveOptimization)]

- <a name="suteu2019orthogonal"></a> Suteu, M., & Guo, Y. 
*[Regularizing Deep Multi-Task Networks using Orthogonal Gradients.](https://arxiv.org/abs/1912.06844)*,
ArXiv, 2019. 

- <a name="yu2020surgery"></a> Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. 
*[Gradient surgery for multi-task learning.](https://arxiv.org/abs/2001.06782)*,
ArXiv, 2020. [[Tensorflow](https://github.com/tianheyu927/PCGrad)]




