# User Behavior Classification Pipeline

## Project Overview

This project explores player behavior in a **synthetic mobile game
dataset** to identify features that distinguish valuable players from
other players. It combines exploratory data analysis, Random Forest
classification, and model interpretation to uncover behavioral patterns
that may inform product hypotheses and future experiments.

## Objectives

-   Explore player behavior and feature distributions.
-   Define a binary target for valuable players using a revenue
    threshold.
-   Train and evaluate a Random Forest classifier.
-   Interpret model behavior using feature importance, Partial
    Dependence Plots (PDPs), and SHAP.
-   Translate findings into potential product questions to investigate
    through controlled experiments.

## Workflow

1.  **Data preparation** --- remove identifiers and unused columns,
    handle missing values, and inspect descriptive statistics.
2.  **Outlier treatment** --- cap extreme numeric values using
    winsorization.
3.  **Target definition** --- classify players with revenue greater than
    `2.8` as valuable.
4.  **Comparison-group sampling** --- retain all valuable players and
    randomly downsample the comparison group to create balanced classes.
5.  **Model training and evaluation** --- train a Random Forest
    classifier and assess F1 score and ROC-AUC.
6.  **Hyperparameter tuning** --- use randomized search with five-fold
    cross-validation.
7.  **Model interpretation** --- examine feature importance, PDPs, and
    SHAP summary and dependence plots.

## Results

After hyperparameter tuning, the model achieved the following results on
the held-out test set:

  Metric       Score
  ---------- -------
  F1 score     0.847
  ROC-AUC      0.876

These metrics describe performance on the **balanced test sample** used
in this analysis; they should not be interpreted as expected performance
on the original player population.

The interpretation plots help identify which features the model uses to
distinguish the two groups and how feature values relate to its
predictions. Findings should be treated as model-based associations, not
evidence that changing a feature will cause revenue to increase.

## Business Relevance

The analysis can help generate hypotheses about player behavior and
monetization. For example, a feature associated with valuable players
may suggest an area for further product investigation. A controlled A/B
test would then be needed to assess whether changing that feature
improves outcomes such as revenue, engagement, or retention.

## Tools and Libraries

-   Python
-   pandas and NumPy
-   scikit-learn
-   Matplotlib and Seaborn
-   SHAP
-   Plotly

The notebook also uses the project helper module
`classification_model_setup.py`.

## Dataset

The project uses synthetic data (`synthetic_users.csv`). The results
illustrate an analytical workflow and should not be interpreted as
findings about real players or a real game.

## Limitations

-   The dataset is synthetic, so the findings are illustrative.
-   The revenue threshold (`revenue > 2.8`) defines the target for this
    analysis.
-   Downsampling creates balanced comparison groups, so reported test
    metrics reflect that sampling design.
-   Feature importance, PDPs, and SHAP explain model behavior; they do
    not establish causation.
-   Product recommendations would require validation with real data and
    controlled experiments.

## Next Steps

-   Investigate the strongest behavioral signals in more detail.
-   Form specific, testable product hypotheses.
-   Validate hypotheses using real game data and A/B experiments.
