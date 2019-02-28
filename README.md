# AI Papers and Notes

## Paper Notes

- [Guided Anchoring](notes/guided_anchoring.md)
- [Cascade R-CNN](notes/cascade_rcnn.md)
- [Hybrid Task Cascade](notes/htc.md)
- [RetinaNet/Focal Loss](notes/retinanet.md)
- [Light-Head R-CNN](notes/lighthead_rcnn.md)
- [Data Distillation](notes/data_distillation.md)

## Read but need to add notes


- [Fast R-CNN](https://arxiv.org/abs/1504.08083). April 2015
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497). June 2015
- [Mask R-CNN](https://arxiv.org/abs/1703.06870). March 2017
- [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240). Nov 2017. Introduced SyncBN
- [[FPN] Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144). Dec 2016.
- [TF-Replicator: Distributed Machine Learning For Researchers](https://arxiv.org/pdf/1902.00465.pdf)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). June 2017. Facebook. ResNet time-to-train SotA. Paper with Linear Scaling Rule: scale LR linearly with batch size
- [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205) July 2018. Tencent. ResNet time-to-train SotA 
- [ImageNet/ResNet-50 Training in 224 Seconds](https://arxiv.org/abs/1811.05233). Nov 2018. Sony. ResNet time-to-train SotA
- [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888). Layer-wise adaptive rate scaling (LARS)
- [[SENet] Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). Sept 2017.
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186). June 2015
- [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489). Nov 2017.
- [[ResNet] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Dec 2015
- [Memory-Efficient Backpropagation Through Time](https://arxiv.org/abs/1606.03401). June 2016
- [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://arxiv.org/abs/1712.02616). Dec 2017
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174.pdf)
- [Group Normalization](https://arxiv.org/abs/1803.08494). March 2018



## Unread Papers

#### High Priority

- [FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](https://arxiv.org/abs/1901.03495). Jan 2019. Best new backbone? Resnet+ accuracy with fewer params. Part of SotA entry COCO Object Detection 2018
- [[Transformer] Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer-XL: Attentive Language Models Beyond A Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). Jan 2019 SotA. Pre-GPT-2. CMU + Google Brain
- [[ODENet] Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf). NeurIPS 2018 best paper, time-series.
- [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/). DeepMind. Starcraft II RL agent that reaches ~95% percentile of human players.
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640). June 2015
- [[YOLOv2] YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242). Dec 2016
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) April 2018
- [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211). June 2017
- [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168). Nov 2018
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409). May 2016. Object detection
- [[PANet] Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534) . March 2018. FPN top-down path + new bottom-up path. [Referenced from HTC paper literature review]
- [DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling](https://arxiv.org/abs/1703.10295). object detection is estimating a very large but extremely sparse bounding box dependent probability distribution



#### Medium Priority


- [[PlanNet] Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) ([blog](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)). Google. A Deep Planning Network for Reinforcement Learning
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf). Google LM, Oct 2018 SotA (pre MT-DNN which uses BERT to set new SotA) 
- [[GPT] Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).  OpenAI LM, June 2018 SotA (pre BERT). Minimal benchmark overlap with ELMo
- [[GPT-2] Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). Feb 2019 SotA, Multi-task NLP
- [[MT-DNN] Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504). , Microsoft. BERT + Multi-task model. Feb 2019 SotA, Multi-task NLP (comparison with GPT-2 unknown) 
- [[NASNet] Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) ([blog](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html)). Google Brain. July 2017.
- [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://arxiv.org/abs/1812.08011). IBM Watson. December 2018
- [LAG: Lazily Aggregated Gradient for Communication-Efficient Distributed Learning](https://arxiv.org/abs/1805.09965). NeurIPs 2018
- [GradiVeQ: Vector Quantization for Bandwidth-Efficient Gradient Aggregation in Distributed CNN Training](https://arxiv.org/abs/1811.03617). NeurIPs 2018
- [Stochastic Gradient Push for Distributed Deep Learning](https://research.fb.com/publications/stochastic-gradient-push-for-distributed-deep-learning/). FAIR, NeurIPs 2018
- [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf). OpenAI
- [You May Not Need Attention](https://arxiv.org/abs/1810.13409)
- [Human-level performance in first-person multiplayer games with population-based deep reinforcement learning](https://arxiv.org/abs/1807.01281) ([blog](https://deepmind.com/blog/capture-the-flag/)). Jul 2018. Have agents play each other
- [CRAFT: Complementary Recommendations Using Adversarial Feature Transformer](https://arxiv.org/abs/1804.10871). Amazon.com. April 2018.
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325). Dec 2015 


#### Low Priority

- [DetNet: A Backbone network for Object Detection](https://arxiv.org/abs/1804.06215). April 2018
- [[ResNeXt] Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431). Nov 2016.
- [[LASER] Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464) ([blog](https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/)). Dec 2018.
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Google. Apr 2017
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Google. Jan 2018.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). May 2015
- [[MVCNN] Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://arxiv.org/abs/1505.00880). May 2015
- [Measuring the Effects of Data Parallelism on Neural Network Training](https://arxiv.org/abs/1811.03600)
- [Optimal Algorithms for Non-Smooth Distributed Optimization in Networks](https://arxiv.org/abs/1806.00291). NeurIPS 2018 best paper 
- [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012). Nov 2016


#### Unprioritized

- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://research.fb.com/publications/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition/). FAIR, CVPR 2018
- [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://research.fb.com/publications/condensenet-an-efficient-densenet-using-learned-group-convolutions/). FAIR, CVPR 2018
- [Deep Spatio-Temporal Random Fields for Efficient Video Segmentation](https://research.fb.com/publications/deep-spatio-temporal-random-fields-for-efficient-video-segmentation/). FAIR, CVPR 2018
- [Detail-Preserving Pooling in Deep Networks](https://research.fb.com/publications/detail-preserving-pooling-in-deep-networks/). FAIR, CVPR 2018
- [Detect-and-Track: Efficient Pose Estimation in Videos](https://research.fb.com/publications/detect-and-track-efficient-pose-estimation-in-videos/). FAIR, CVPR 2018
- [Detecting and Recognizing Human-Object Interactions](https://research.fb.com/publications/detecting-and-recognizing-human-object-interactions/). FAIR, CVPR 2018
- [LAMV: Learning to align and match videos with kernelized temporal layers](https://research.fb.com/publications/lamv-learning-to-align-and-match-videos-with-kernelized-temporal-layers/). FAIR, CVPR 2018
- [Learning to Segment Every Thing](https://research.fb.com/publications/learning-to-segment-every-thing/). FAIR, CVPR 2018
- [What Makes a Video a Video: Analyzing Temporal Information in Video Understanding Models and Datasets](https://research.fb.com/publications/what-makes-a-video-a-video-analyzing-temporal-information-in-video-understanding-models-and-datasets/). FAIR, CVPR 2018
- [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/pdf/1901.09005.pdf). Google Brain
- [Self-Driving Cars: A Survey](https://arxiv.org/pdf/1901.04407.pdf)
- [Non-delusional Q-learning and Value Iteration](https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration.pdf). NeurIPS 2018 best paper
- [Learning Unsupervised Learning Rules](https://arxiv.org/pdf/1804.00222.pdf). Google Brain, meta-learning, mid 2018
- [Spherical CNNs](https://openreview.net/pdf?id=Hkbd5xZRb). ICLR 2018 Best Paper
- [Continuous Adaptation Via Meta-Learing In Nonstationary And Competitive Environments](https://openreview.net/pdf?id=Sk2u1g-0-). ICLR 2018 Best Paper
- [On the Convergence of Adam and Beyond ](https://openreview.net/forum?id=ryQu7f-RZ). ICLR 2018 Best Paper
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://www.nyu.edu/projects/bowman/glue.pdf)
- [[DecaNLP] The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730). Benchmark task
- [[ELMo] Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf). Mar 2018 SotA (pre BERT). Minimal benchmark overlap with GPT.
- [Universal Transformers](https://arxiv.org/abs/1807.03819). July 2018
- [[MS-CNN] A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection](https://arxiv.org/abs/1607.07155). Multi-scale object detection. FPN related?
- [[AttractioNet] Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization](https://arxiv.org/abs/1606.04446). Multi-stage procedure for generating accurate object proposals. Attend-and-refine module to update bounding box locations iteratively. [References from HTC paper literature review]  
- [CRAFT Objects from Images](https://arxiv.org/abs/1604.03239). Incorporates cascade structure into RPN. [References from HTC paper literature review]
- [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/pdf/1810.04020.pdf)
- [[Instance-FCN] Instance-sensitive Fully Convolutional Networks](https://arxiv.org/abs/1603.08678). Detection based, instance segmentation. [References from HTC paper literature review]
- [[MNC] Instance-aware Semantic Segmentation via Multi-task Network Cascades](https://arxiv.org/abs/1512.04412) Dec 2015. Detection based, instance segmentation. Three subtasks (instance localization, mask prediction, object categorization) trained in cascade manner [References from HTC paper literature review]
- [[FCIS] Fully Convolutional Instance-aware Semantic Segmentation](https://arxiv.org/abs/1611.07709) Nov 2016. Detection based, instance segmentation. Extends InstanceFCN. Fully convolutional [References from HTC paper literature review]
- MaskLab, DeepMask, SharpMask (detection based, instance segmentation) [References from HTC paper literature review]
- Multi-region CNN (iteratize localization mechanism that alternates between box scoring and location refinement) [References from HTC paper literature review]
- IoUNet (performs progressive bounding box refinement, even though not presenting a cascade structure explicitly) [References from HTC paper literature review]
- CC-Net (rejects easy RoIs at shallow layers) [References from HTC paper literature review]
- A convolutional neural network cascade for face detection (proposes to operate at multiple resolutions to reject simple samples) [Referenced from HTC paper literature review]
- [COCO-Stuff: Thing and Stuff Classes in Context](https://arxiv.org/pdf/1612.03716.pdf). COCO dataset augmentation with pixelwise stuff annotations. Things are classes like 'cars', 'bikes'. Stuff is :'sky', 'grass'.
- Atrous Spatial Pyramid Pooling (ASPP)
- Global Convolutional Network (GCN)
- SoftNMS 
- [[A3C] Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) DeepMind. RL.
- D4PG (RL)
- [StarCraft AI Competitions, Bots and Tournament Manager Software](https://www.researchgate.net/publication/329202945_StarCraft_AI_Competitions_Bots_and_Tournament_Manager_Software)
- [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782). The Starcraft II learning environment
- [Deep reinforcement learning with relational inductive biases](https://openreview.net/forum?id=HkxaFoC9KQ). DeepMind. ICLR 2019. Starcraft II minigame SotA.
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926). Mid-late 2017
- [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832.pdf). DeepMind. Nov 2017
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561). DeepMind. Feb 2018.
- [Self-Imitation Learning](https://arxiv.org/abs/1806.05635). RL. Google Brain. June 2018. "a simple off-policy actor-critic algorithm"
- [Generative Adversarial Self-Imitation Learning](https://arxiv.org/abs/1812.00950) Dec 2018. Google Brain
- [POLICY DISTILLATION](https://arxiv.org/pdf/1511.06295.pdf). DeepMind. Jan 2016.
- [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643). DeepMind. NeurIPS 2018. RL.
- [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) ([blog](https://deepmind.com/blog/alphazero-shedding-new-light-grand-games-chess-shogi-and-go])). Chess, self-playing
- AlphaFold ([blog](https://deepmind.com/blog/alphafold/))
- [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941). Feb 2018. 
- [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443). Dec 2018