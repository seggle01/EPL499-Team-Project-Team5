import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(y_test, y_pred, target_names=['negative', 'positive']):
    """
    Print evaluation metrics for a classification model.
    
    Displays accuracy score and detailed classification report including precision,
    recall, and F1-score for each class [web:37][web:39].
    
    Parameters
    ----------
    y_test : array-like
        True labels from the test set.
    y_pred : array-like
        Predicted labels from the model.
    target_names : list, optional
        List of target class names for display (default: ['negative', 'positive']).
    
    Returns
    -------
    None
        Prints metrics to console.
    """
    # Print section header for clarity
    print("\n=== Model Evaluation ===")
    
    # Calculate and display overall accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Generate detailed classification report with per-class metrics
    print(classification_report(y_test, y_pred, target_names=target_names))

def plot_confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive']):
    """
    Plot confusion matrix heatmap to visualize classification performance.
    
    Creates a heatmap showing the counts of true positives, false positives,
    true negatives, and false negatives [web:37][web:40].
    
    Parameters
    ----------
    y_test : array-like
        True labels from the test set.
    y_pred : array-like
        Predicted labels from the model.
    labels : list, optional
        List of label names for axis display (default: ['Negative', 'Positive']).
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    # Compute confusion matrix from true and predicted labels
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure with appropriate size
    plt.figure(figsize=(8, 6))
    
    # Generate heatmap with annotations showing counts
    sns.heatmap(
        cm,
        annot=True,        # Display numeric values in cells
        fmt='d',           # Format as integers
        cmap='Blues',      # Blue color scheme
        xticklabels=labels,
        yticklabels=labels
    )
    
    # Add axis labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def analyze_misclassifications(y_test, y_pred, df_test):
    """
    Analyze and report misclassified examples in detail.
    
    Identifies all misclassified instances and displays their text content
    along with true and predicted labels for manual inspection.
    
    Parameters
    ----------
    y_test : pandas.Series
        True labels from the test set with index preserved.
    y_pred : array-like
        Predicted labels from the model.
    df_test : pandas.DataFrame
        Testing dataset containing 'text' column with tweet content.
    
    Returns
    -------
    None
        Prints misclassification analysis to console.
    """
    # Find indices where predictions don't match true labels
    misclassified_indices = y_test.index[y_test != y_pred].tolist()

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Total Misclassified: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
    print(f"{'='*80}\n")

    # Create mapping from integer labels to human-readable names
    int_to_label = {1: 'Positive', 0: 'Negative'}

    # Display first 50 misclassified examples for manual review
    for i, idx in enumerate(misclassified_indices[:50], 1):  # Show first 50
        # Get true and predicted labels for this example
        true_label = y_test[idx]
        pred_label = y_pred[y_test.index.get_loc(idx)]
        
        # Retrieve original tweet text
        text = df_test.loc[idx, 'text']
        
        # Display formatted example with separator
        print(f"Example {i}:")
        print(f"  Text: {text}")
        print(f"  True Label: {int_to_label[true_label]}")
        print(f"  Predicted: {int_to_label[pred_label]}")
        print(f"  {'-'*76}\n")

def get_feature_importance(X_train, y_train, top_n=25, random_state=42):
    """
    Calculate and display feature importance using Random Forest classifier.
    
    Trains a Random Forest model and extracts feature importance scores to identify
    which features contribute most to classification decisions [web:40].
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features with column names.
    y_train : array-like
        Training labels.
    top_n : int, optional
        Number of top features to display (default: 25).
    random_state : int, optional
        Random state for reproducibility (default: 42).
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'feature' and 'importance' columns, sorted by importance
        in descending order.
    """
    # Train Random Forest to calculate feature importances
    model_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model_rf.fit(X_train, y_train)
    
    # Extract feature names and their importance scores
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model_rf.feature_importances_
    })
    
    # Sort by importance in descending order
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    # Display top N features in console
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importances.head(top_n))
    
    return feature_importances

def plot_feature_importance(feature_importances, top_n=25):
    """
    Plot feature importance as a horizontal bar chart.
    
    Visualizes the most important features from a trained model to understand
    which features drive predictions.
    
    Parameters
    ----------
    feature_importances : pandas.DataFrame
        DataFrame with 'feature' and 'importance' columns from get_feature_importance().
    top_n : int, optional
        Number of top features to plot (default: 25).
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    # Create figure with appropriate size for readability
    plt.figure(figsize=(10, 8))
    
    # Select top N most important features
    top_features = feature_importances.head(top_n)
    
    # Create horizontal bar chart with importance values
    plt.barh(range(len(top_features)), top_features['importance'].values, color='skyblue')
    
    # Set y-axis labels to feature names
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    
    # Add axis labels and title
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    
    # Invert y-axis so highest importance is at top
    plt.gca().invert_yaxis()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
