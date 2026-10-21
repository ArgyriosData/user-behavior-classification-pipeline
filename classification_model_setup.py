import pandas as pd
import numpy as np


def analyze_multiple_stats(data, columns=None):
    import pandas as pd
    
    if columns is None:
        columns = data.columns  # default to all columns
    
    results = []
    for col in columns:
        stats = {
            'Feature': col,
            'Min': data[col].min(),
            'Max': data[col].max(),
            'Mean': data[col].mean(),
            'Median': data[col].median(),
            '25th Percentile': data[col].quantile(0.25),
            '75th Percentile': data[col].quantile(0.75),
            '95th Percentile': data[col].quantile(0.95),
            '99th Percentile': data[col].quantile(0.99),
            '999th Percentile': data[col].quantile(0.999)
        }
        results.append(pd.DataFrame([stats]))
    
    return pd.concat(results, ignore_index=True)





def winsorize_df(df, lower=0.01, upper=0.99):
    df_wins = df.copy()
    for col in df_wins.select_dtypes(include=[np.number]).columns:
        low_val = df_wins[col].quantile(lower)
        high_val = df_wins[col].quantile(upper)
        df_wins[col] = np.clip(df_wins[col], low_val, high_val)
    return df_wins





import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# def plot_partial_dependence_grid(rf, X_train_top, top_features, export_dir, zoom_fraction=0.3):
def plot_partial_dependence_grid(rf, X_train_top, top_features, zoom_fraction=0.3):

    """
    Plot partial dependence for a list of features with zoomed-in comparison,
    save side-by-side plots to export_dir, and display them.

    Parameters
    ----------
    rf : estimator
        Trained sklearn model (e.g., RandomForestClassifier).
    X_train_top : pd.DataFrame
        Training data with selected features.
    top_features : list
        List of feature names to plot.
    export_dir : str
        Directory path where plots will be saved.
    zoom_fraction : float, optional
        Fraction of x-axis range to zoom in (default=0.3).
    """

    # os.makedirs(export_dir, exist_ok=True)

    for feature in top_features:
        # Full plot
        fig1, ax = plt.subplots()
        disp = PartialDependenceDisplay.from_estimator(rf, X_train_top, [feature], ax=ax)
        for ax in disp.axes_.ravel():
            ax.set_ylabel("Predicted probability")
        ax.set_title(f"Partial Dependence:\n {feature}")
        fig1.tight_layout()

        # Zoomed plot
        fig2, ax = plt.subplots()
        disp = PartialDependenceDisplay.from_estimator(rf, X_train_top, [feature], ax=ax)
        xmin, xmax = plt.xlim()
        zoom_max = xmin + zoom_fraction * (xmax - xmin)
        plt.xlim(xmin, zoom_max)
        for ax in disp.axes_.ravel():
            ax.set_ylabel("Predicted probability")
        plt.title(f'Partial Dependence (Zoomed):\n {feature}')
        fig2.tight_layout()

        # Convert figures to images
        fig1.canvas.draw()
        fig2.canvas.draw()
        img1 = np.array(fig1.canvas.renderer.buffer_rgba())
        img2 = np.array(fig2.canvas.renderer.buffer_rgba())
        plt.close(fig1)
        plt.close(fig2)

        # Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img1); axes[0].axis("off")
        axes[1].imshow(img2); axes[1].axis("off")
        plt.tight_layout()

        # Export
        # save_path = os.path.join(export_dir, f"partial_dependence_{feature}.jpeg")
        # fig.savefig(save_path, format="jpeg", dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)


from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Train predictions
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    # Test predictions
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("Train F1:", f1_score(y_train, y_pred_train))
    print("Train ROC-AUC:", roc_auc_score(y_train, y_proba_train))
    print("Test F1:", f1_score(y_test, y_pred_test))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_proba_test))

