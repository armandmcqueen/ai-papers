# [Guided Anchoring](https://arxiv.org/abs/1901.03278)

([pdf](https://arxiv.org/pdf/1901.03278.pdf))

- Jan 2019
- SenseTime Joint Lab (The Chinese University of Hong Kong), Amazon Rekognition and Nanyang Technological University
- For RPN based models, learn how to generate anchors instead of using the basic sliding window approach. 
- Improves recall of the RPN
- Has a cost to training time and a cost to RPN inference time, but improves accuracy (across multiple tasks) 
- Use fine-tuning to add GA-RPN to a model training with a classic RPN (3 additional epochs)
- May improve inference time when the model has a heavy head (Mask RCNN?)
