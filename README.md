# Deep Learning Fundamentals — Tutorial Portfolio

**Course:** CMPE 258 — Deep Learning  
**Semester:** Spring 2026
** Name ** Kalhar Mayurbhai Patel (019140511)

A hands-on portfolio of six deep learning tutorials built in Google Colab. Each notebook goes from zero to hero on a core topic — activation functions, CNNs, hyperparameter tuning, classification metrics, modern architectures, and optimizers — with working code, visualizations, and detailed explanations.

---

## Quick Links — Colabs & Video Walkthroughs

| # | Tutorial | Colab Notebook | Video Walkthrough |
|:-:|----------|:--------------:|:-----------------:|
| 1 | Activation Functions | [Open in Colab](https://colab.research.google.com/drive/1X4Y6crmtXKSrVAfSZ0Rpsos0IlSfhUwz?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |
| 2 | CNN Fundamentals | [Open in Colab](https://colab.research.google.com/drive/1eGNGpbdstT8PEvmrNge_oWWnQR9t_PHp?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |
| 3 | Hyperparameter Tuning | [Open in Colab](https://colab.research.google.com/drive/1IXOj5VBlCjDWQdwjjhA0FfLdwBdCc1Vc?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |
| 4 | Classification Metrics | [Open in Colab](https://colab.research.google.com/drive/1MgNvpp3txkDM7tQrIVd-hvi7I-Fy39YV?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |
| 5 | Modern CNN Architectures | [Open in Colab](https://colab.research.google.com/drive/1kkICFrcHwIJAGOvNtfiqEYohHe1KOris?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |
| 6 | Optimizers for Deep Learning | [Open in Colab](https://colab.research.google.com/drive/1lXcXt5KnPE4Y3hFAS-dhgzltmAgqeooD?usp=sharing) | [▶ YouTube](TODO_ADD_LINK) |

> **📌 Replace each `TODO_ADD_LINK` with your actual YouTube URL after recording.**

---

## Notebook 1 — Activation Functions for Deep Learning

**File:** `final_activation_functions_tutorial.ipynb` · 9 code cells · 11 markdown sections

This notebook answers the question *"why can't a neural network just be a stack of linear layers?"* and then systematically introduces every activation function you'll encounter in practice.

**Part I — Why Activation Functions Matter.** The notebook opens by proving mathematically that stacking linear layers without activations collapses into a single linear transformation (W₂·W₁·x = W·x). It then visualizes three problems that linear models can't solve: XOR, concentric circles, and spiral data — all of which require non-linear decision boundaries.

**Part II — Classic Activations: Sigmoid & Tanh.** Sigmoid (σ(x) = 1/(1+e⁻ˣ)) and Tanh are implemented from scratch in NumPy. The code plots each function alongside its derivative, showing how sigmoid's gradient maxes out at 0.25 and tanh's at 1.0 — both saturate for large inputs, causing the vanishing gradient problem in deep networks.

**Part III — The ReLU Revolution.** ReLU (max(0,x)) is plotted with its derivative (a step function). The notebook explains why ReLU transformed deep learning in the 2010s: constant gradient of 1 for positive inputs, computationally cheap, and sparse activations. It then demonstrates the "dying ReLU" problem — neurons that output zero for all inputs and stop learning entirely.

**Part IV — ReLU Variants.** Implements Leaky ReLU (small slope α=0.01 for negatives), PReLU (learnable α), ELU (smooth exponential curve for negatives), and SELU (self-normalizing with specific α and scale constants). Each variant is plotted side-by-side with its derivative, showing how they address the dying ReLU problem differently.

**Part V — Modern Activations.** Covers GELU (used in BERT/GPT — a smooth approximation that probabilistically gates inputs), Swish/SiLU (x·sigmoid(x) — discovered by neural architecture search at Google), and Mish (x·tanh(softplus(x))). The code overlays all three on one plot, highlighting how they allow small negative values through unlike ReLU.

**Part VI — Output Layer Activations.** Explains that output activations depend on the task: sigmoid for binary classification (outputs probability 0–1), softmax for multi-class (outputs probability distribution summing to 1), and no activation (linear) for regression.

**Part VII — Practical Decision Guide.** Provides a flowchart: use ReLU as default → GELU for transformers → SiLU for computer vision → SELU for self-normalizing networks.

**Part VIII — Hands-on Comparison.** Trains identical PyTorch networks on the moons dataset with 8 different activations and compares training loss curves, final accuracy, and decision boundaries side-by-side. The results confirm that modern activations (GELU, SiLU, Mish) converge faster and produce smoother decision boundaries than ReLU on this task.

---

## Notebook 2 — Convolutional Neural Networks from First Principles

**File:** `final_cnn_fundamentals_tutorial.ipynb` · 26 code cells · 9 markdown sections

Builds CNNs from the ground up — starting with what a convolution actually computes, all the way through training a full image classifier on CIFAR-10.

**Chapter 1 — The Convolution Operation.** Opens with the parameter explosion problem: a 224×224 RGB image fed into a fully connected layer with 1000 neurons requires 150 million weights. The code implements 2D convolution manually with nested loops, applies it to a sample image, and visualizes the result. Classic edge-detection kernels (Sobel horizontal, Sobel vertical, Laplacian) are applied to show how different 3×3 filters extract different features.

**Chapter 2 — Building CNN Layers in PyTorch.** Introduces `nn.Conv2d` with detailed parameter breakdowns (in_channels, out_channels, kernel_size, stride, padding). The code visualizes the output size formula: O = (I − K + 2P) / S + 1. It demonstrates padding modes (same vs valid) and stride effects with before/after feature map visualizations. A multi-channel convolution section shows how a Conv2d layer processes all input channels simultaneously and produces multiple output channels.

**Chapter 3 — Pooling.** Explains why we pool (reduce spatial size, increase receptive field, add translation invariance). Implements and compares MaxPool2d, AvgPool2d, and AdaptiveAvgPool2d. Visualizes the effect of each pooling type on a real image, showing how max pooling preserves sharp edges while average pooling produces smoother results.

**Chapter 4 — Complete CNN Architectures.** Implements LeNet-5 (the 1998 historic architecture) in PyTorch, then builds a modern CNN with batch normalization, dropout, and the Conv→BN→ReLU→Pool pattern. A feature map size visualization traces how spatial dimensions shrink and channel count grows through each layer.

**Chapter 5 — Training CNNs.** Implements a full data augmentation pipeline (random horizontal flip, rotation, color jitter, random crop) using `torchvision.transforms` and visualizes augmented versions of the same image. Loads CIFAR-10, sets up a training loop with cross-entropy loss and Adam optimizer, and trains for multiple epochs with validation tracking.

**Chapter 6 — Visualizing What CNNs Learn.** Extracts and plots the learned convolutional filters from layer 1 (shows edge/color detectors the network discovered on its own). Then passes an image through the network and visualizes intermediate feature maps at each layer — showing the progression from edges → textures → high-level parts.

**Chapter 7 — Production-Ready Classifier.** Builds a more sophisticated CNN with residual-style connections, trains it on CIFAR-10 with cosine annealing, and produces a full evaluation: per-class accuracy, confusion matrix heatmap, and misclassified example analysis.

---

## Notebook 3 — Hyperparameter Tuning for Deep Learning

**File:** `final_hyperparameter_tuning_tutorial.ipynb` · 10 code cells · 9 markdown sections

A practical guide to finding optimal hyperparameters, progressing from manual tuning through automated Bayesian optimization.

**Part I — Parameters vs. Hyperparameters.** Explains the distinction: parameters (weights, biases) are learned during training; hyperparameters (learning rate, batch size, hidden layer sizes, dropout rate) are set before training. The code demonstrates the impact by training the same architecture on the moons dataset with different learning rates and hidden sizes, plotting how dramatically results change.

**Part II — Key Hyperparameters.** Presents a ranked table of the most important hyperparameters by category (optimization, architecture, regularization) with typical ranges. Visualizes the search space — learning rate spans several orders of magnitude (1e-5 to 1e-1), batch sizes are typically powers of 2, and dropout ranges 0.1–0.5.

**Part III — Manual Tuning: The Learning Rate Finder.** Implements a learning rate finder that gradually increases LR from 1e-7 to 1 while recording the loss. Identifies the "sweet spot" just before the loss explodes and plots the characteristic LR-finder curve.

**Part IV — Grid Search.** Implements exhaustive grid search over learning rate × hidden size combinations. Trains a model for each combination and visualizes results as a heatmap. Discusses the curse of dimensionality: adding more hyperparameters makes the grid explode exponentially.

**Part V — Random Search.** Implements random sampling from continuous distributions. Side-by-side visualization shows why random search explores more unique values of each hyperparameter compared to grid search with the same budget. Cites the Bergstra & Bengio (2012) result that random search is more efficient because typically only a few hyperparameters truly matter.

**Part VI — Bayesian Optimization.** Explains the surrogate model approach: build a probabilistic model of the objective, use it to predict promising regions, balance exploration vs. exploitation, iterate.

**Part VII — Optuna Tutorial.** Implements a complete Optuna study with `trial.suggest_float` and `trial.suggest_int`. Optuna runs trials with TPE sampling. The code visualizes the optimization history and parameter importance plots.

**Part VIII — Best Practices.** Practical workflow: start simple → LR finder → tune most important params first → use random/Bayesian search → always use a validation set, never tune on test.

---

## Notebook 4 — Classification Metrics: A Complete Guide

**File:** `final_important_classification_metrics_tutorial.ipynb` · 62 code cells · 99 markdown sections

The most comprehensive notebook in this collection — a full course organized as 18 progressive tasks covering every classification and regression metric used in practice.

**Task 1 — Setup & The Classification Problem.** Creates a synthetic binary classification dataset with controlled class imbalance, visualizes the distribution, performs train-test split, and trains a Logistic Regression baseline used throughout all remaining tasks.

**Task 2 — The Confusion Matrix.** Defines and visualizes the four outcomes (TP, TN, FP, FN) with color-coded cells. Explains Type I errors (false positives — "false alarm") and Type II errors (false negatives — "missed detection") using the courtroom analogy. Analyzes specific misclassified examples.

**Task 3 — Accuracy & Its Limitations.** Computes accuracy and demonstrates the "accuracy paradox": on a 95/5 imbalanced dataset, a model predicting only the majority class gets 95% accuracy while catching zero positive cases. Introduces balanced accuracy.

**Task 4 — Precision & Recall.** Precision = TP/(TP+FP) — "when I predict positive, am I right?" Recall = TP/(TP+FN) — "did I find all the positives?" The code varies the threshold from 0 to 1 and plots the precision-recall trade-off. Real-world guidance: prioritize precision for spam detection, recall for cancer screening.

**Task 5 — F1, F2, and Fβ Scores.** Explains why the harmonic mean penalizes extreme imbalance (a model with 100% precision but 1% recall gets F1=0.02, not 50.5%). Implements the Fβ family. Also covers Matthews Correlation Coefficient (MCC) and Cohen's Kappa.

**Task 6 — ROC Curves & AUC.** Plots the ROC curve (TPR vs FPR at every threshold). Explains AUC: 0.5 = random, 1.0 = perfect. Demonstrates the probabilistic interpretation — AUC is the probability that a random positive scores higher than a random negative.

**Task 7 — Precision-Recall Curves.** Explains when PR curves beat ROC: on imbalanced datasets, ROC can look optimistic. PR curves focus on positive-class performance. Computes Average Precision (AP).

**Task 8 — Multi-class Metrics.** Extends binary metrics using macro, micro, and weighted averaging. Shows when each is appropriate.

**Task 9 — Imbalanced Datasets.** Identifies metrics that work under imbalance (PR-AUC, F1, MCC, balanced accuracy) vs. metrics to avoid (raw accuracy).

**Task 10 — MLOps Pipeline Metrics.** Covers metrics at each ML lifecycle stage (development, validation, deployment, monitoring). Builds a comprehensive metrics dashboard.

**Task 11 — Multi-label Metrics.** Hamming loss, subset accuracy, Jaccard score, and label-based precision/recall/F1 for multi-label classification.

**Task 12 — Semi-supervised Metrics.** Pseudo-label quality evaluation, transductive vs. inductive assessment, self-training demonstration.

**Task 13 — Confidence & Thresholds.** Confidence calibration (reliability diagrams, Expected Calibration Error), optimal threshold selection, confidence bucketization, and human-in-the-loop routing.

**Tasks 14–18 — Regression Metrics.** MAE, MSE, RMSE (Task 15), R² and Adjusted R² (Task 16), MAPE (Task 17), and industry best-practices for metric selection by domain — finance, healthcare, retail, energy (Task 18).

---

## Notebook 5 — Modern CNN Architectures: ResNet to EfficientNet

**File:** `final_modern_cnn_architectures_tutorial.ipynb` · 18 code cells · 8 markdown sections

Picks up where Notebook 2 left off, covering the architectural breakthroughs from 2015+ that enabled 100+ layer networks.

**Chapter 1 — The Vanishing Gradient Problem.** Builds plain networks of increasing depth (8, 16, 32, 64 layers) and plots gradient magnitude at each layer during backpropagation. Gradients decay exponentially — by layer 50, they're effectively zero and early layers stop learning.

**Chapter 2 — ResNet: The Residual Revolution.** Implements the basic residual block: instead of learning H(x), learn the residual F(x) = H(x) − x, output F(x) + x. The skip connection lets gradients flow directly through the identity path. Builds both the basic block (two 3×3 convs) and the bottleneck block (1×1 → 3×3 → 1×1 used in ResNet-50+). Assembles a complete ResNet and shows via gradient flow analysis that residual networks maintain healthy gradient magnitudes across all layers.

**Chapter 3 — Advanced Building Blocks.** Implements Squeeze-and-Excitation (SE) blocks — a channel attention mechanism: global average pooling → FC → ReLU → FC → sigmoid to weight channels by importance. Then implements Mobile Inverted Bottleneck (MBConv) — depthwise separable convolutions (depthwise 3×3 per channel + pointwise 1×1 to mix channels) to drastically reduce parameters.

**Chapter 4 — EfficientNet & Compound Scaling.** The three scaling dimensions (width, depth, resolution) and EfficientNet's key insight: scaling all three simultaneously with fixed ratios beats scaling any single one. Loads pretrained EfficientNet-B0 and prints its architecture.

**Chapter 5 — Transfer Learning.** Two approaches: feature extraction (freeze backbone, train new head) vs. fine-tuning (unfreeze and retrain). Loads pretrained EfficientNet, replaces the classifier for CIFAR-10's 10 classes, prepares data with ImageNet normalization.

**Chapter 6 — Fine-Tuning Strategies.** Implements discriminative learning rates (lower LR for early layers, higher for later layers and new head) via separate parameter groups. Trains both approaches and compares convergence and accuracy side-by-side.

---

## Notebook 6 — Optimizers for Deep Learning

**File:** `final_optimizers_deep_learning_tutorial.ipynb` · 21 code cells · 23 markdown sections

The richest notebook for visualizations — packed with 3D surface plots, contour maps, and optimizer trajectory comparisons.

**Part I — Why Optimizers Matter.** Frames optimization as the central problem of deep learning. Visualizes loss landscapes in 1D, 2D, and 3D. Shows four landscape types: simple bowl, ravine (elongated valley), saddle point, and multimodal (multiple minima) via contour plots.

**Part II — Gradient Descent Fundamentals.** Explains gradients and the update rule W_new = W_old − η·∇L. Implements vanilla GD from scratch, runs it on a 2D quadratic, and plots the trajectory on a contour map. Demonstrates the learning rate's impact with six cases: too small (crawls), just right (converges), too large (oscillates or diverges).

**Part III — Problems with Vanilla GD.** Two failure modes visualized: (1) The ravine problem — on L = w₁² + 10·w₂², GD oscillates across the narrow direction instead of following the valley. (2) Saddle points — on L = w₁² − w₂², gradient is zero at origin but it's not a minimum; vanilla GD gets stuck.

**Part IV — Stochastic Gradient Descent.** Compares batch GD (smooth but slow) vs. SGD (noisy but can escape local minima). Visualizes both on a multimodal landscape. Discusses batch size trade-offs by domain.

**Part V — Momentum Methods.** Implements SGD+Momentum from scratch. The physics analogy — a rolling ball accumulates velocity. Shows momentum solving the ravine problem by dampening oscillations. Then implements Nesterov Accelerated Gradient (NAG) — compute gradient at the "look-ahead" position.

**Part VI — Adaptive Learning Rate Methods.** Implements AdaGrad (per-parameter LR scaling from accumulated squared gradients — good for sparse data but LR decays to zero). Then RMSprop (Hinton's fix: exponential moving average instead of accumulation). Each optimizer's trajectory is plotted on the ravine landscape.

**Part VII — Adam & AdamW.** Implements Adam from scratch: first moment (momentum), second moment (RMSprop), plus bias correction. Explains why Adam is the default. Then the AdamW distinction: standard L2 regularization in Adam gets scaled by the adaptive rate, weakening it. AdamW decouples weight decay from gradient adaptation.

**Part VIII — Learning Rate Schedules.** Visualizes six schedules: step decay, exponential decay, cosine annealing, warm restarts, linear warmup, and one-cycle policy.

**Part IX — Practical PyTorch Comparison.** Trains the same network with five optimizers (SGD, SGD+Momentum, RMSprop, Adam, AdamW) and plots loss, accuracy, and convergence side-by-side. Produces a decision flowchart: fine-tuning → AdamW; best generalization → SGD+Momentum+schedule; default → Adam.

**Part X — Modern & Advanced Optimizers.** Covers SAM (Sharpness-Aware Minimization — seeks flat minima by maximizing loss in a neighborhood, then minimizing). Visualizes sharp vs. flat minima. Closes with the optimizer evolution timeline from 1847 (Cauchy) to 2020 (SAM).

---

## Repository Structure

```
deep-learning-tutorials/
├── README.md
├── notebooks/
│   ├── final_activation_functions_tutorial.ipynb       (9 code cells)
│   ├── final_cnn_fundamentals_tutorial.ipynb            (26 code cells)
│   ├── final_hyperparameter_tuning_tutorial.ipynb       (10 code cells)
│   ├── final_important_classification_metrics_tutorial.ipynb  (62 code cells)
│   ├── final_modern_cnn_architectures_tutorial.ipynb    (18 code cells)
│   └── final_optimizers_deep_learning_tutorial.ipynb    (21 code cells)
```

---

## How to Run

1. Click any **Open in Colab** link in the table above.
2. Go to **Runtime → Change runtime type → T4 GPU** (recommended for CNN and optimizer notebooks).
3. **Runtime → Run all** to execute every cell.
4. Dependencies install automatically via `!pip install -q ...` in the first cell of each notebook.

---

## Technologies Used

- **Python 3** — all notebooks
- **PyTorch** — neural network construction, training, and optimization
- **NumPy** — numerical computation and manual algorithm implementations
- **Matplotlib** — all visualizations (2D plots, 3D surfaces, contour maps, heatmaps)
- **scikit-learn** — datasets (moons, circles), metrics, train/test splitting
- **Optuna** — Bayesian hyperparameter optimization (notebook 3)
- **torchvision** — pretrained models, data augmentation transforms, CIFAR-10 dataset

---

## Assignment Checklist

- [ ] All 6 notebooks copied to Google Drive with world-sharing permissions
- [ ] All 6 notebooks executed in Colab with outputs saved
- [ ] All 6 notebooks pushed to this GitHub repository
- [ ] 6 walkthrough videos recorded (one per notebook, code-block by code-block)
- [ ] 6 videos uploaded to YouTube
- [ ] YouTube links added to the Quick Links table above (replace `TODO_ADD_LINK`)
- [ ] This README pushed to repo root
