"""
Model evaluation and visualization utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def evaluate_model(y_test, y_pred, target_names=['negative', 'positive']):
    """
    Print evaluation metrics for a classification model.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        target_names: List of target class names
    """
    print("\n=== Model Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))

def plot_confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive']):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        labels: List of label names for display
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def analyze_misclassifications(y_test, y_pred, df_test):
    """
    Analyze and report misclassified examples.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        df_test: Testing dataset
    """
    misclassified_indices = y_test.index[y_test != y_pred].tolist()

    print(f"\n{'='*80}")
    print(f"Total Misclassified: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
    print(f"{'='*80}\n")

    # Display misclassified tweets
    int_to_label = {1: 'Positive', 0: 'Negative'}

    for i, idx in enumerate(misclassified_indices[:50], 1):  # Show first 50
        true_label = y_test[idx]
        pred_label = y_pred[y_test.index.get_loc(idx)]
        text = df_test.loc[idx, 'text']
        
        print(f"Example {i}:")
        print(f"  Text: {text}")
        print(f"  True Label: {int_to_label[true_label]}")
        print(f"  Predicted: {int_to_label[pred_label]}")
        print(f"  {'-'*76}\n")

def get_feature_importance(X_train, y_train, top_n=25, random_state=42):
    """
    Calculate and display feature importance using Random Forest.
    
    Args:
        X_train: Training features
        y_train: Training labels
        top_n: Number of top features to display
        random_state: Random state for reproducibility
        
    Returns:
        feature_importances: DataFrame with feature names and importance scores
    """
    model_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model_rf.fit(X_train, y_train)
    
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model_rf.feature_importances_
    })
    
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importances.head(top_n))
    
    return feature_importances

def plot_feature_importance(feature_importances, top_n=25):
    """
    Plot feature importance bar chart.
    
    Args:
        feature_importances: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
    """
    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'].values, color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()