# AI Lab Assistant 2025 Recruitment Task 4
Reinforcement Learning

---

<!-- CONTRIBUTOR -->
 <div align="center" id="contributor">
   <strong>
     <h3> Contributors </h3>
     <table align="center">
       <tr align="center">
         <td>NIM</td>
         <td>Name</td>
         <td>GitHub</td>
       </tr>
       <tr align="center">
         <td>13523004</td>
         <td>Razi Rachman Widyadhana</td>
         <td align="center" >
           <div style="margin-right: 20px;">
           <a href="https://github.com/zirachw" ><img src="https://avatars.githubusercontent.com/u/148220821?v=4" width="48px;" alt=""/> 
             <br/> <sub><b> @zirachw </b></sub></a><br/>
           </div>
         </td>
       </tr>
     </table>
   </strong>
 </div>

---

## Supervised Learning (Bagian 2)
- [X] **KNN** - K-Nearest Neighbors
- [X] **LR** - Logistic Regression  
- [x] **Gaussian Naive Bayes** - Classifier probabilistik dengan implementasi manual
- [x] **CART** - Decision Tree (Classification and Regression Trees)
- [x] **SVM** - Support Vector Machine dengan multiple kernel (linear, RBF, polynomial, sigmoid)
- [x] **ANN** - Artificial Neural Network dengan arsitektur layer modular

**Bonus yang diimplementasikan:**
- **Newton's Method** - Alternatif optimisasi untuk estimasi parameter model
- **Multiple SVM Kernels** - Kernel RBF, polynomial, dan sigmoid dengan quadratic programming
- **Adam Optimizer** - Optimisasi lanjutan untuk neural network
- **CNN Architecture** - 2D Convolution Layer dan 2D Max Pooling Layer untuk arsitektur LeNet

## Unsupervised Learning (Bagian 3)
- [x] **K-MEANS** - Algoritma clustering dengan inisialisasi K-means++
- [x] **DBSCAN** - Density-based clustering dengan multiple distance metrics
- [x] **PCA** - Principal Component Analysis untuk dimensionality reduction

**Bonus yang diimplementasikan:**
- **K-means++** - Improved centroid initialization untuk clustering yang lebih baik

## Reinforcement Learning (Bagian 4)
- [X] **Q-Learning** - Model-free reinforcement learning
- [x] **SARSA** - On-policy temporal difference learning

## Instalasi & Setup

### Prerequisites
- Python 3.13+
- uv (modern Python package manager)

### Instalasi
```bash
# Install dependencies
uv sync

# Pilih kernel uv di Notebook, terus run all :D
```

### Dependencies Utama
- **Core**: `numpy`, `pandas`, `matplotlib`, `seaborn`
- **ML Libraries**: `scikit-learn`, `tensorflow`
- **Optimization**: `cvxopt` (untuk SVM)
- **Data Processing**: `scipy`
