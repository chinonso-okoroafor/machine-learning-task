
---

# 🧮 MATH501: Modelling & Analytics for Data Science — End-to-End Statistical & Machine Learning Mastery  
> *From Seismic Event Classification to Airline Satisfaction & Carbon Sequestration — A Showcase of Statistical Rigor, Model Selection, and Business-Aligned Analytics*

![R](https://img.shields.io/badge/R-Advanced-blue?logo=r)  
![Python](https://img.shields.io/badge/Python-Supplementary-lightgrey?logo=python)  
![Machine Learning](https://img.shields.io/badge/ML-Decision%20Trees%20+%20KNN-orange)  
![Bayesian](https://img.shields.io/badge/Bayesian-JAGS%20+%20MCMC-purple)  
![Stats](https://img.shields.io/badge/Statistics-ANOVA%20+%20Tukey%20HSD-red)  
![Clustering](https://img.shields.io/badge/Unsupervised-KMeans-green)

## 📊 Project 1: Machine Learning — Seismic Event Classification (Earthquake vs Explosion)

### 🎯 Business Problem
Classify seismic events with high accuracy and interpretability to support emergency response and national security decisions.

### 🔍 Data Exploration
- Dataset: 37 observations, 2 features (`Body`, `Surface`), binary label (`type`: earthquake/explosion)
- **Imbalanced**: 26 earthquakes, 11 explosions
- **Key Insight**: Explosions have ↑ `Body`, ↓ `Surface` — clear separation with moderate correlation (r=0.2)

![Scatterplot](screenshots/scatter_body_surface.png)  
*Fig: Explosions (red) cluster in high-Body, low-Surface region — ideal for classification.*

---

### 🤖 Model Development & Tuning

#### ➤ K-Nearest Neighbors (KNN)
- **Tuned `k` via LOOCV** → Optimal `k=3` (validation error = 0)
- **Test Error (LOOCV)**: 5.4%
- **Boundary**: Radial, smooth — less interpretable

![KNN Boundary](screenshots/knn_boundary.png)

#### ➤ Decision Tree
- **Optimal Size**: 3 terminal nodes (via cross-validation)
- **Rules**:  
  `IF Body < 5.675 AND Surface < 4.44 → Explosion`  
  `ELSE → Earthquake`
- **Test Error (LOOCV)**: **0%** — perfect classification
- **Why Chosen**: Zero error + full interpretability → critical for high-risk decisions

![Decision Tree Rules](screenshots/decision_tree_rules.png)

---

### 📈 Model Comparison & Selection

| Metric               | Decision Tree | KNN     |
|----------------------|---------------|---------|
| **Test Error (LOOCV)** | **0.000**     | 0.054   |
| **False Positive Rate** | **0.000**     | 0.077   |
| **Interpretability**   | **High**      | Low     |
| **Recommended For**    | Mission-critical systems | General classification |

> ✅ **Final Selection**: Decision Tree — balances accuracy, simplicity, and risk mitigation.

---

### 🔍 Bonus: Unsupervised Learning — K-Means Clustering
- **Goal**: Discover hidden structure without labels
- **Method**: Elbow Method → Optimal `k=3`
- **Insight**: Cluster 3 ≈ “Explosions”; Clusters 1 & 2 = subtypes of “Earthquakes” (high vs low Body/Surface)

![K-Means Clusters](screenshots/kmeans_clusters.png)

---

## 📈 Project 2: Frequentist Statistics — Airline Customer Satisfaction Analysis

### 🎯 Business Question
Which airlines have statistically different customer satisfaction scores? Is Airline D >3 points better than B or C?

### 📊 Data Summary
- 4 Airlines (A, B, C, D), 15 customers each → Balanced design
- Key Stats:  
  - **D**: Mean=6.33 (Highest)  
  - **B**: Mean=5.67  
  - **A & C**: ~4.4

![Boxplot](screenshots/airline_boxplot.png)

---

### 📉 Statistical Testing

#### ➤ One-Way ANOVA
- **H₀**: All means equal → **Rejected** (F=5.29, p=0.00278)  
  → At least one airline differs significantly.

#### ➤ Tukey HSD (Pairwise Comparisons)
| Comparison | Diff | p-value   | Significant? |
|------------|------|-----------|--------------|
| **D vs A** | 2.00 | **0.007** | ✅ Yes       |
| **D vs C** | 1.87 | **0.014** | ✅ Yes       |
| **D vs B** | 0.67 | 0.676     | ❌ No        |
| **B vs A** | 1.33 | 0.123     | ❌ No        |

#### ➤ Hypothesis Test: Is D >3 points better than B or C?
- **D - B ≤ 3?** → p=1.0 → **No evidence D is >3 points better**  
- **D - C ≤ 3?** → p=0.994 → **No evidence D is >3 points better**

> 💡 **Business Insight**: Airline D outperforms A and C — but not by 3+ points vs B or C. Marketing should avoid overclaiming.

---

## 🌱 Project 3: Bayesian Statistics — Carbon Sequestration in Agriculture

### 🎯 Business Problem
Which carbon capture treatment (T1-T5) is most effective? Do field locations (1-3) matter?

### 📊 Data Structure
- 3 Fields × 5 Treatments → 15 observations
- Response: Total Carbon Sequestration

---

### 📉 Bayesian Two-Way ANOVA (JAGS + MCMC)

#### ➤ Model 1: Full Model (Field + Treatment Effects)
```r
yᵢⱼ ~ N(μ + αᵢ + βⱼ, σ²)
α₁=0 (Field 1 baseline), β₁=0 (T1 baseline)
```
- **Field Effects (α)**:  
  - α₂=1.73 (95% HPD: -8.49, 11.32) → Not significant (includes 0)  
  - α₃=1.32 (95% HPD: -9.02, 10.97) → Not significant  
  → **Conclusion**: Field location has no detectable effect.

- **Treatment Effects (β)**:  
  - **T4**: β₄=30.67 (95% HPD: 18.32, 43.70) → **Strongest effect**  
  - T3: β₃=21.66, T5: β₅=18.05, T2: β₂=12.83 → All positive & significant (HPD >0)  
  → **Conclusion**: T4 > T3 > T5 > T2 > T1

![Caterpillar Plot - Treatments](screenshots/caterpillar_treatments.png)

- **Model Fit**: DIC=119.5

---

### ➤ Model 2: Simpler Model (Treatment Only)
```r
yᵢⱼ ~ N(μ + βⱼ, σ²)  // No field effect
```
- **Treatment Effects**: Same ranking (T4 strongest)
- **Model Fit**: **DIC=108.6** → Better fit + simpler → **Preferred model**

> ✅ **Farmer’s Decision**: Use **Treatment T4** — maximizes carbon capture. Field choice doesn’t matter.

---

## 🛠️ Technical Stack & Skills Demonstrated

| Area                  | Tools & Techniques                                                                 | Business Value                                  |
|-----------------------|------------------------------------------------------------------------------------|------------------------------------------------|
| **Machine Learning**  | Decision Trees, KNN, Hyperparameter Tuning (LOOCV), Confusion Matrices             | High-accuracy, interpretable classification    |
| **Frequentist Stats** | ANOVA, Tukey HSD, Linear Hypothesis Testing, p-values, Effect Sizes                | Validated comparisons, prevented false claims  |
| **Bayesian Stats**    | JAGS, MCMC, Prior/Posterior, HPD Intervals, DIC, Trace/Density Plots               | Quantified uncertainty, model comparison       |
| **Unsupervised**      | K-Means, Elbow Method, Cluster Interpretation                                      | Discovered hidden patterns in unlabeled data   |
| **Data Viz**          | ggplot2, Boxplots, Scatterplots, Histograms, Caterpillar Plots, Scree Plots         | Communicated insights to stakeholders          |
| **Programming**       | R (Advanced), `tree`, `e1071`, `aov`, `TukeyHSD`, `R2jags`, `ggplot2`, `dplyr`     | Reproducible, production-ready analysis        |
| **Critical Thinking** | Model Justification, Result Interpretation, Limitations, Business Context          | Trusted advisor, strategic decision support    |

---

## 📁 Repository Structure

```
├── MATH501_Report.pdf              # Full coursework report
├── code/
│   ├── seismic_classification.R    # ML: Decision Tree + KNN + Clustering
│   ├── airline_anova.R             # Frequentist: ANOVA + Tukey + Hypothesis Tests
│   └── carbon_bayesian.R           # Bayesian: JAGS Models + MCMC Diagnostics
├── data/
│   ├── earthquake.txt              # Seismic event data
│   ├── airline_satisfaction.csv    # Airline customer scores
│   └── carbon_sequestration.csv    # Agricultural treatment data
├── screenshots/                    # Validation images (add your PNGs here)
│   ├── scatter_body_surface.png
│   ├── decision_tree_rules.png
│   ├── airline_boxplot.png
│   ├── caterpillar_treatments.png
│   └── kmeans_clusters.png
└── README.md                       # You are here!
```

---

## ▶️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/math501-modelling-analytics.git
   cd math501-modelling-analytics
   ```

2. Install R dependencies:
   ```r
   install.packages(c("tree", "e1071", "car", "multcomp", "R2jags", "ggplot2", "dplyr"))
   ```

3. Run scripts in RStudio or R console:
   ```r
   source("code/seismic_classification.R")
   source("code/airline_anova.R")
   source("code/carbon_bayesian.R")
   ```

## 📚 References

- Ripley, B. (2023). `tree`: Classification and Regression Trees.  
- Meyer, D. et al. (2023). `e1071`: Misc Functions for Statistics.  
- Su, Y. & Yajima, M. (2024). `R2jags`: Bayesian Analysis with JAGS.  
- Wickham, H. (2016). `ggplot2`: Elegant Graphics for Data Analysis.  
- Venables, W.N. & Ripley, B.D. (2002). *Modern Applied Statistics with S*.

---

## 🤝 Connect & Collaborate

👤 **Author**: [Your Name]  
📧 **Email**: [your.email@example.com]  
💼 **LinkedIn**: [linkedin.com/in/yourprofile]  
🎓 **Program**: MSc Data Science and Business Analytics, University of Plymouth

> 👉 *Open to roles in: Statistical Modeling, Research Science, Advanced Analytics, Risk Modeling, Public Sector Data Science.*

---

✅ **Last Updated**: April 2024  
✅ **License**: MIT — Use, adapt, learn, and build upon this work!

---
