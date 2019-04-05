# AI Papers and Notes

## Paper Notes

- [Guided Anchoring](notes/guided_anchoring.md)
- [Cascade R-CNN](notes/cascade_rcnn.md)
- [Hybrid Task Cascade](notes/htc.md)
- [RetinaNet/Focal Loss](notes/retinanet.md)
- [Light-Head R-CNN](notes/lighthead_rcnn.md)
- [Data Distillation](notes/data_distillation.md)
- [YOLO v1](notes/yolo.md)

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
- [[C3D] Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)
- [[OFF] Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)


## Unread Papers

#### High Priority

- [Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241v1.pdf). Mar 2019. Horizon Robotics. Mask RCNN + scoring block that learns quality of predictions. Beats MaskRCNN accuracy.
- [FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](https://arxiv.org/abs/1901.03495). Jan 2019. Best new backbone? Resnet+ accuracy with fewer params. Part of SotA entry COCO Object Detection 2018
- [[Transformer] Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer-XL: Attentive Language Models Beyond A Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). Jan 2019 SotA. Pre-GPT-2. CMU + Google Brain
- [[ODENet] Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf). NeurIPS 2018 best paper, time-series.
- [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/). DeepMind. Starcraft II RL agent that reaches ~95% percentile of human players.
- [[YOLOv2] YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242). Dec 2016
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) April 2018
- [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211). June 2017
- [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168). Nov 2018
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409). May 2016. Object detection
- [[PANet] Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534) . March 2018. FPN top-down path + new bottom-up path. [Referenced from HTC paper literature review]
- [DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling](https://arxiv.org/abs/1703.10295). object detection is estimating a very large but extremely sparse bounding box dependent probability distribution
- [Speeding up Deep Learning with Transient Servers](https://arxiv.org/pdf/1903.00045v1.pdf). Feb 2019. WPI, Chinese Academy of Sciences
- [A Structured Model For Action Detection](https://arxiv.org/pdf/1812.03544v3.pdf). Feb 2019. CMU, Google. SotA on AVA dataset (according to abstract.)
- [Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling](https://arxiv.org/pdf/1902.08295v1.pdf). Feb 2019. Google (presumably). https://github.com/tensorflow/lingvo
- [FIXUP INITIALIZATION: RESIDUAL LEARNING WITHOUT NORMALIZATION](https://arxiv.org/pdf/1901.09321v1.pdf). Jan 2019. Work done at Facebook.
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/pdf/1901.01892.pdf). Mar 2019. University of Chinese Academy of Sciences, TuSimple. SotA COCO object detection
- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383v1.pdf). Nov 2018. MIT



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
- [FEDERATED LEARNING FOR MOBILE KEYBOARD PREDICTION](https://arxiv.org/pdf/1811.03604v2.pdf). Feb 2019. Google.
- [ConvNet Architecture Search for Spatiotemporal Feature Learning](https://arxiv.org/pdf/1708.05038.pdf). Aug 2017. FB + Columbia
- [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/pdf/1712.04851.pdf) July 2018. Google + UCSD
- [50 Years of Test (Un)fairness: Lessons for Machine Learning](https://arxiv.org/pdf/1811.10104v2.pdf). Dec 2018. Google
- [Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques and Tools](https://arxiv.org/pdf/1903.11314v1.pdf). Mar 2019. Technical University of Munich
- [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/pdf/1903.10955v2.pdf). Mar 2019. CUHK + SenseTime + University Sydney + Beihang University
- [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/pdf/1903.11027v1.pdf). Mar 2019. nuTonomy
- [DetNAS: Neural Architecture Search on Object Detection](https://arxiv.org/pdf/1903.10979v1.pdf). Mar 2019. Chinese Academy of Science + Megvii
- [Large-scale interactive object segmentation with human annotators](https://arxiv.org/pdf/1903.10830v1.pdf). Mar 2019. Google
- [TensorFlow Eager: A Multi-Stage, Python-Embedded DSL for Machine Learning](https://arxiv.org/abs/1903.01855). Feb 2019. Google Brain

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
- [Random Search and Reproducibility for Neural Architecture Search](https://arxiv.org/abs/1902.07638). Feb 2019. CMU
- [Deep Object-Centric Policies for Autonomous Driving](https://arxiv.org/pdf/1811.05432v2.pdf). Mar 2019. UC Berkeley
- [Accelerating Self-Play Learning in Go](https://arxiv.org/pdf/1902.10565v2.pdf). Mar 2019. Jane Street
- [Representation Flow for Action Recognition](https://arxiv.org/pdf/1810.01455v2.pdf). Mar 2019, Indiana University. Abstract claim advantage in computational speed and 'performance'
- [Multimodal Trajectory Predictions for Autonomous Driving using Deep Convolutional Networks](https://arxiv.org/pdf/1809.10732v2.pdf). Mar 2019. Uber ATG
- [Continuous Integration of Machine Learning Models with ease.ml/ci: Towards a Rigorous Yet Practical Treatment](https://arxiv.org/pdf/1903.00278v1.pdf). Mar 2019. Many authors: ETH Zurich, Alibaba Group, Huawei Technologies, Modulos AG, Microsoft Research
- [Object Recognition in Deep Convolutional Neural Networks is Fundamentally Different to That in Humans](https://arxiv.org/pdf/1903.00258v1.pdf). Mar 2019. University of Aberdeen, University of Essex. psychology dept.
- [Double Quantization for Communication-Efficient Distributed Optimization](https://arxiv.org/pdf/1805.10111v3.pdf). Mar 2019. Tsinghua University, Tencent AI 
- [RESTRUCTURING BATCH NORMALIZATION TO ACCELERATE CNN TRAINING](https://arxiv.org/pdf/1807.01702v2.pdf). Mar 2019. Seoul National University and Samsung. SysML
- [Characterizing Activity on the Deep and Dark Web](https://arxiv.org/pdf/1903.00156v1.pdf). Mar 2019. USC, Georgia Tech. Interesting data set.
- [Video Extrapolation with an Invertible Linear Embedding](https://arxiv.org/pdf/1903.00133v1.pdf). Mar 2019. BYU. Predict future video frames.
- [Video Summarization via Actionness Ranking](https://arxiv.org/pdf/1903.00110v1.pdf). Mar 2019. University of Central Florida, Center for Research in Computer Vision (CRCV).
- [Actor and Action Video Segmentation from a Sentence](https://arxiv.org/pdf/1803.07485v1.pdf). Mar 2018. University of Amsterdam
- [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/pdf/1902.10250v1.pdf). Feb 2019. UC Berkeley
- [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909v1.pdf). Feb 2019. Alibaba.
- [Efficient Video Classification Using Fewer Frames](https://arxiv.org/pdf/1902.10640v1.pdf). Feb 2019. Indian Institute of Technology Madras, NVIDIA Bangalore
- [An End-to-End Network for Panoptic Segmentation](https://arxiv.org/pdf/1903.05027v2.pdf). Mar 2019. Zhejiang University, Megvii. 'Promising results' on COCO Panoptic
- [Two-Stream Oriented Video Super-Resolution for Action Recognition](https://arxiv.org/pdf/1903.05577v1.pdf). Mar 2019. University of Science and Technology of China
- [SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition](https://arxiv.org/pdf/1903.05831v1.pdf). Mar 2019. MXNet (w/ extra C++ ops) object detection development framework.
- [Deep learning for time series classification: a review](https://arxiv.org/pdf/1809.04356v3.pdf). Mar 2019. 
- [ADAPTIVE COMMUNICATION STRATEGIES TO ACHIEVE THE BEST ERROR-RUNTIME TRADE-OFF IN LOCAL-UPDATE SGD](https://arxiv.org/pdf/1810.08313v2.pdf). Mar 2019. SysML. CMU
- [Real time backbone for semantic segmentation](https://arxiv.org/pdf/1903.06922v1.pdf). Mar 2019. 
- [Learning Correspondence from the Cycle-consistency of Time](https://arxiv.org/pdf/1903.07593v1.pdf). Mar 2019. CMU + UC-Berkeley "We introduce a self-supervised method for learning visual correspondence from unlabeled video"
- [Scaling Human Activity Recognition to edge devices](https://arxiv.org/pdf/1903.07563v1.pdf). Mar 2019. UCSD. Looks at I3D and TSM
- [Understanding the Limitations of CNN-based Absolute Camera Pose Regression](https://arxiv.org/pdf/1903.07504v1.pdf). Mar 2019. Chalmers University of Technology, TU Munich, ETH Zurich, Microsoft
- [IVANET: LEARNING TO JOINTLY DETECT AND SEGMENT OBJETS WITH THE HELP OF LOCAL TOP-DOWN MODULES](https://arxiv.org/pdf/1903.07360v1.pdf). Mar 2019. Northeastern University (of Shenyang, Liaoning province, China)
- [In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images](https://arxiv.org/pdf/1903.08469v1.pdf). Mar 2019. University of Zagreb
- [Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/pdf/1903.07785v1.pdf). Mar 2019. FAIR. "We present a new approach for pretraining a bi-directional transformer model that provides significant performance gains across a variety of language understanding problems"
- [TICTAC: ACCELERATING DISTRIBUTED DEEP LEARNING WITH COMMUNICATION SCHEDULING](https://arxiv.org/abs/1803.03288). Mar 2018. SysML 2019. University of Illinois at UrbanaChampaign
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/pdf/1806.03377.pdf). June 2018. Microsoft Research + Carnegie Mellon University + Stanford University
- [PRIORITY-BASED PARAMETER PROPAGATION FOR DISTRIBUTED DNN TRAINING](https://www.sysml.cc/doc/2019/75.pdf). SysML 2019. University of British Columbia + Vector Institute + CMU + University of Toronto
- [BLUECONNECT: DECOMPOSING ALL-REDUCE FOR DEEP LEARNING ON HETEROGENEOUS NETWORK HIERARCHY](https://www.sysml.cc/doc/2019/130.pdf). SysML 2019. IBM
- [BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS](https://www.sysml.cc/doc/2019/16.pdf). SysML 2019. Stanford (Matei Zaharia)
- [3LC: Lightweight and Effective Traffic Compression for Distributed Machine Learning](https://arxiv.org/abs/1802.07389). Feb 2018. SysML 2019. CMU + Intel
- [CATDET: CASCADED TRACKED DETECTOR FOR EFFICIENT OBJECT DETECTION FROM VIDEO](https://arxiv.org/abs/1810.00434). Sep 2018. SysML 2019. Stanford + NVIDIA
- [AdaScale: Towards Real-time Video Object Detection Using Adaptive Scaling](https://arxiv.org/abs/1902.02910). Feb 2019. SysML 2019. CMU
- [Restructuring Batch Normalization to Accelerate CNN Training](https://arxiv.org/abs/1807.01702). March 2019. SysML 2019. Seoul National University + Samsung
- [Bandana: Using Non-volatile Memory for Storing Deep Learning Models](https://arxiv.org/abs/1811.05922). Nov 2018. SysML 2019. Stanford + Facebook
- [Mini-batch Serialization: CNN Training with Inter-layer Data Reuse](https://arxiv.org/abs/1810.00307). Sep 2018. SysML 2019. UT Austin + UMich + Duke
- [DATA VALIDATION FOR MACHINE LEARNING](https://www.sysml.cc/doc/2019/167.pdf). SysML 2019. Google + KAIST
- [KERNEL MACHINES THAT ADAPT TO GPUS FOR EFFECTIVE LARGE BATCH TRAINING](https://arxiv.org/abs/1806.06144). June 2018. SysML 2019. Ohio State
- [Looking Fast and Slow: Memory-Guided Mobile Video Object Detection](https://arxiv.org/pdf/1903.10172v1.pdf). Mar 2019. Cornell + Google
- [Adversarial Joint Image and Pose Distribution Learning for Camera Pose Regression and Refinement](https://arxiv.org/pdf/1903.06646v2.pdf). Mar 2019. Technical University of Munich + Siemens AG + Johns Hopkins University
- [Spiking-YOLO: Spiking Neural Network for Real-time Object Detection](https://arxiv.org/pdf/1903.06530v1.pdf). Mar 2019.  Seoul National University
- [Few-Shot Learning-Based Human Activity Recognition](https://arxiv.org/pdf/1903.10416v1.pdf). Mar 2019. UMass Amherst
- [Fast Interactive Object Annotation with Curve-GCN](https://arxiv.org/pdf/1903.06874v1.pdf). Mar 2019. University of Toronto + Vector Institute + NVIDIA
- [BLVD: Building A Large-scale 5D Semantics Benchmark for Autonomous Driving](https://arxiv.org/pdf/1903.06405v1.pdf). Mar 2019.  Xian Jiaotong University + Chang’an University
- [Two-Stream Oriented Video Super-Resolution for Action Recognition](https://arxiv.org/pdf/1903.05577v1.pdf). Mar 2019. University of Science and Technology of China
- [Activation Analysis of a Byte-Based Deep Neural Network for Malware Classification](https://arxiv.org/pdf/1903.04717v2.pdf). Mar 2019. FireEye
- [ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector](https://arxiv.org/pdf/1804.05810v2.pdf). Sep 2018. Georgia Tech + Intel
- [Deep Learning on Graphs: A Survey](https://arxiv.org/pdf/1812.04202v1.pdf) Dec 2018. Tsinghua University
- [Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them](https://arxiv.org/pdf/1903.03862v1.pdf). Mar 2019. Bar-Ilan University + Allen Institute for Artificial Intelligence
- [Improving image classifiers for small datasets by learning rate adaptations](https://arxiv.org/pdf/1903.10726v2.pdf). Mar 2019. UTokyo + exMedio Inc. Faster (and therefore cheaper) training
- [Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving](https://arxiv.org/pdf/1903.11444v1.pdf). Mar 2019. Dalin University of Technology + University of Sydney
- [Reducing the dilution: An analysis of the information sensitiveness of capsule network with a practical solution](https://arxiv.org/pdf/1903.10588v2.pdf). Mar 2019. hust.edu.cn
- [Self-Supervised Learning via Conditional Motion Propagation](https://arxiv.org/pdf/1903.11412v1.pdf). Mar 2019. CUHK - SenseTime Joint Lab, Nanyang Technological University
- [Deep Learning based Pedestrian Detection at Distance in Smart Cities](https://arxiv.org/pdf/1812.00876v2.pdf). Northumbria University + Imam Mohammed ibn Saud Islamic University + Lancaster
- [Hearing your touch: A new acoustic side channel on smartphones](https://arxiv.org/pdf/1903.11137v1.pdf). Mar 2019. Soundwaves from tapping can be used to determine values being typed 
- [Simple Applications of BERT for Ad Hoc Document Retrieval](https://arxiv.org/pdf/1903.10972v1.pdf). Mar 2019. University of Waterloo
- [In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images](https://arxiv.org/pdf/1903.08469v1.pdf). Mar 2019. University of Zagreb
- [LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/pdf/1903.08701v1.pdf). Mar 2019. Uber ATG
- [All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification](https://arxiv.org/pdf/1903.05285v1.pdf). Mar 2019. Hikvision Research Institute
- [[FE-Net] Progressive Sparse Local Attention for Video Object Detection](https://arxiv.org/pdf/1903.09126v2.pdf). Mar 2019. NLPR,CASIA + Horizon Robotics
- [[OANet] An End-to-End Network for Panoptic Segmentation](https://arxiv.org/pdf/1903.05027v2.pdf). Mar 2019. Zhejiang University + Megvii Inc. (Face++) + Huazhong University of Science and Technology + Peking University + The University of Tokyo