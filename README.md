### README for Movie Recommendation System Using SVT and BPR

#### Project Overview
This project implements a **Movie Recommendation System** using two advanced techniques: **Singular Value Thresholding (SVT)** and **Bayesian Personalized Ranking (BPR)**. These approaches aim to enhance the recommendation process by utilizing matrix completion and pairwise ranking optimization for explicit and implicit feedback, respectively.

---

#### Features
1. **Singular Value Thresholding (SVT)**:
   - Utilizes matrix factorization for recovering missing values in a sparse user-item interaction matrix.
   - Implements an iterative algorithm to reconstruct the matrix with minimal computational cost.
   - Ideal for explicit feedback, such as user ratings.

2. **Bayesian Personalized Ranking (BPR)**:
   - Optimizes personalized rankings using implicit feedback data like clicks and purchases.
   - Employs a pairwise ranking approach to ensure relevant items are ranked higher.
   - Utilizes Stochastic Gradient Descent (SGD) for scalable and efficient optimization.

---

#### Dataset
1. **Ratings.csv**: Contains user ratings with columns:
   - `userId`, `movieId`, `rating`, `timestamp`.
2. **Movies.csv**: Contains movie details with columns:
   - `movieId`, `title`, `genres`.

---

#### Requirements
- Python 3.8+
- Libraries: `numpy`, `pandas`, `sklearn`, `tkinter`.

---

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NishanthSaravanamurali/Recommendation_systems-SVT_BPR.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Recommendation_systems-SVT_BPR
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

#### How to Run
1. Ensure `Ratings.csv` and `Movies.csv` files are in the specified directory.
2. Run the **SVT Recommendation System**:
   ```bash
   python svt_recommender.py
   ```
3. Run the **BPR Recommendation System**:
   ```bash
   python bpr_recommender.py
   ```

---

#### Key Algorithms
1. **SVT Implementation**:
   - Performs matrix completion by applying singular value decomposition (SVD) iteratively.
   - Outputs a low-rank reconstructed user-item matrix for accurate recommendations.

2. **BPR Implementation**:
   - Uses a latent factor model to embed users and items in a shared vector space.
   - Optimizes the ranking of items using pairwise comparisons of observed vs. unobserved interactions.

---

#### Performance
- **SVT**: Suitable for datasets with explicit feedback. 
- **BPR**: Best for datasets with implicit feedback, ensuring high scalability.

---

#### Comparison
| Feature                 | SVT                                | BPR                                |
|-------------------------|-------------------------------------|------------------------------------|
| Feedback Type           | Explicit (ratings)                | Implicit (clicks, views, purchases)|
| Optimization Objective  | Matrix reconstruction              | Pairwise ranking                  |
| Computational Efficiency| Moderate                          | High                              |

---

#### References
- [Singular Value Thresholding](https://ww3.math.ucla.edu/camreport/cam08-77.pdf)
- [Bayesian Personalized Ranking](https://towardsdatascience.com/recommender-system-bayesian-personalized-ranking-from-implicit-feedback-78684bfcddf6)

---

This README serves as a comprehensive guide for setting up and running the movie recommendation system using SVT and BPR approaches. For detailed mathematical formulations, refer to the `MFC_Final.docx`.
