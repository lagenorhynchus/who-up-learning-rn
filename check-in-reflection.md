# Intro: (from proposal)

Linsley, Feng, and Serre (2025) show that there is a decoupling between DNN performance on benchmarks and the extent to which they mirror V1 and IT. The analysis demonstrates that early DNNs increasingly mirrored activity in the primate visual cortex (V1 and the inferior temporal, or IT) as their accuracy improved. However, this trend has reversed in contemporary frontier models. These systems frequently rely on "shortcut learning": exploiting false statistical regularities rather than employing the causal, shape-biased mechanistic reasoning characteristic of biological systems. We aim to address this problem by implementing a CNN that mimics the architecture of the human visual cortex. We want to split the network's processing into two distinct pathways: a simulated magnocellular stream and a parvocellular stream. For context, the magnocellular stream processes motion and where an object is, and the parvocellular stream processes shape and what an object is. The M-stream will be fed low-pass-filtered grayscale image variants to induce learning of global structure, spatial relationships, and contrast. The P-stream will receive high-pass filtered, high-resolution color inputs to process fine details.
This architecture would introduce a direct mechanism for human intervention and explainability. Rather than allowing the model to fuse these features arbitrarily, we will engineer adjustable fusion weights in the network's terminal layers. Manually weighting the magnocellular stream might induce the model to prioritize shape over texture. We want to test the model on both standard training datasets (e.g., CIFAR-100 or ImageNet) and out-of-distribution evaluation datasets designed to trick standard models (such as Stylized-ImageNet). We define success by the dual-stream model maintaining high classification accuracy on stylized images by relying on its shape-focused M-stream.

# Challenges:

We found some of the oscar logistics frustrating but are working through them, as well as coordinating different development branches on github but all good now.

# Insights so far:

Model Architecture:

Magnocellular: 4-stage Conv-BN-ReLU (channels 1→32→64→128→128, stride-2 downsampling) + Global Average Pooling → 128-d feature.

Parvocellular: 3-block Conv-BN-ReLU (channels 3→32→64→128, MaxPool) + Adaptive Average Pooling → 128-d feature.

Fusion: fused = α · m_feat + (1 − α) · p_feat.

With learnable_alpha=True, α is sigmoid-bounded and trained end-to-end.

Classifier: Linear(128 → num_classes).

![alt text](<PNG image.png>)
Experiment 3 (best result) details
Started with alpha=0.8 (shape bias)
Learned to adjust alpha down to 0.29 (still prefers texture)
Achieved highest validation accuracy (34.6%)
Smallest overfitting gap among learnable runs
Training accuracy also highest (44.8%)

# Plan:

Use the Stylized-ImageNet for more results, since we used CIFAR-100
Configure oscar to run this^
