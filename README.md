# Enhanced Convolutional Neural Networks for Image Classification

## 📋 Project Overview

This project implements and extends the research paper **"Enhanced Convolutional Neural Networks for Improved Image Classification"** by Xiaoran Yang, Shuhan Yu, and Wenxi Xu (2023). We reproduce the original methodology and propose several architectural enhancements to improve image classification performance on CIFAR-10 and CIFAR-100 datasets.

## 🎯 Objectives

- **Phase 1**: Implement and reproduce the methodology from the original paper
- **Phase 2**: Extend the work with novel architectural improvements and evaluate their effectiveness

## 🏗️ Architecture Implementations

### 1. Baseline CNN (3-Block)
- **Description**: Direct implementation of the original paper's architecture
- **Structure**: 3 convolutional blocks with batch normalization, ReLU activation, and max pooling
- **CIFAR-100 Accuracy**: 68.01%

### 2. Enhanced 5-Block CNN
- **Description**: Extended network depth with 5 convolutional blocks
- **Channel Progression**: 64 → 128 → 256 → 512 → 512
- **CIFAR-100 Accuracy**: 92.93%
- **Improvement**: +24.92% over baseline

### 3. CNN + Graph Convolutional Network (GCN)
- **Description**: Integration of GCN layers after CNN feature extraction
- **Innovation**: Models feature relationships as graph structures
- **CIFAR-100 Accuracy**: 67.66%
- **Key Features**:
  - Identity-based adjacency matrix
  - Normalized graph convolution
  - Feature-level relationship modeling

### 4. CNN + Squeeze-and-Excitation + Residual Connections
- **Description**: Channel attention mechanism with residual learning
- **CIFAR-100 Accuracy**: 92.78%
- **Key Features**:
  - Channel-wise attention recalibration
  - Skip connections for gradient flow
  - Enhanced feature discrimination

### 5. Residual Attention Network
- **Description**: Trunk-and-mask attention mechanism with residual connections
- **CIFAR-100 Accuracy**: 93.34% ⭐ **Best Performance**
- **Key Features**:
  - Spatial attention masks
  - Trunk-mask parallel processing
  - Deep network with stable training

## 📊 Results Summary

| Architecture | CIFAR-100 Accuracy | Improvement | Key Innovation |
|---|---|---|---|
| Baseline CNN (Paper) | 68.01% | - | Original implementation |
| 5-Block CNN | 92.93% | +24.92% | Increased depth |
| 5-Block + GCN | 67.66% | -0.35% | Graph-based learning |
| SE + Residual CNN | 92.78% | +24.77% | Channel attention |
| Residual Attention | **93.34%** | **+25.33%** | Spatial attention |

## 🔧 Technical Implementation

### Key Components

#### Graph Convolutional Network Layer
```python
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        A = torch.eye(x.size(1), device=x.device)
        A_hat = A + torch.eye(x.size(1), device=x.device)
        D_hat = torch.diag(A_hat.sum(1))
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))
        norm_A = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        x = torch.matmul(x, norm_A)
        return self.linear(x)
```

#### Residual SE Block
```python
class ResidualSEBlock(nn.Module):
    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Squeeze-and-Excitation
        out += identity     # Residual connection
        return self.relu(out)
```

#### Attention Module
```python
class AttentionModule(nn.Module):
    def forward(self, x):
        trunk = self.trunk_branch(x)    # Feature processing
        mask = self.softmax_branch(x)   # Attention mask
        out = trunk * mask + trunk      # Attention application
        return out
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-entropy
- **Batch Size**: 128
- **Epochs**: 100
- **Regularization**: Dropout (0.25-0.5), Early stopping

### Dataset Preparation
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## 📈 Key Findings

### ✅ What Worked Well
1. **Network Depth**: Increasing from 3 to 5 blocks significantly improved performance
2. **Attention Mechanisms**: Both channel (SE) and spatial (Residual Attention) attention were highly effective
3. **Residual Connections**: Essential for training stability and gradient flow

### ⚠️ Challenges Encountered
1. **GCN Integration**: Simple identity-based graphs didn't capture meaningful feature relationships
2. **Overfitting**: Deeper networks required careful regularization
3. **Computational Complexity**: Attention mechanisms increased training time

### 🔍 Insights
- **Residual Attention Network** achieved the best performance (93.34%) but showed signs of overfitting
- **SE + Residual** approach provided excellent balance between performance and generalization
- **Graph-based approaches** need more sophisticated adjacency matrix design

## 🔮 Future Work

### Immediate Improvements
- [ ] Implement more sophisticated graph structures for GCN
- [ ] Add stronger regularization techniques
- [ ] Explore hybrid architectures combining multiple attention mechanisms

### Advanced Extensions
- [ ] Transfer learning evaluation on other datasets
- [ ] Model compression and efficiency optimization
- [ ] Integration with modern architectures (Vision Transformers)
- [ ] Multi-scale attention mechanisms

## 📚 References

1. Yang, X., Yu, S., & Xu, W. (2023). Enhanced Convolutional Neural Networks for Improved Image Classification.
2. He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
3. Hu, J., et al. (2018). Squeeze-and-excitation networks. CVPR.
4. Wang, F., et al. (2017). Residual attention network for image classification. CVPR.
5. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.

## 👥 Team

- Malaika Saleem
- Aiza Ali

## 📄 License

This project is for educational and research purposes.

---

**Note**: This implementation demonstrates various CNN enhancement techniques and their comparative effectiveness on standard benchmark datasets. The code serves as a foundation for further research in deep learning architectures.
