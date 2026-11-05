
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score)

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Feature importance
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
plt.style.use('ggplot')
colors = plt.cm.hsv(np.linspace(0, 1, 6))

def load_and_prepare_data():
    """Load wine quality dataset and prepare for classification."""
    
    print("="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    # Load local wine data
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    
    print(f"\nDataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Check for missing values
    print(f"\nMissing values: {data.isnull().sum().sum()}")
    
    # Convert quality to binary classification (good wine vs bad wine)
    # Quality >= 6 is good (1), < 6 is bad (0)
    data['quality_binary'] = (data['quality'] >= 6).astype(int)
    
    print(f"\nQuality distribution:")
    print(data['quality'].value_counts().sort_index())
    print(f"\nBinary quality distribution:")
    print(data['quality_binary'].value_counts())
    print(f"Class balance: {data['quality_binary'].value_counts(normalize=True)}")
    
    # Separate features and target
    X = data.drop(['quality', 'quality_binary'], axis=1)
    y = data['quality_binary']
    
    return X, y, data


def explore_data(X, y):
    """Perform exploratory data analysis."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Statistical summary
    print("\nFeature statistics:")
    print(X.describe())
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nCorrelation matrix saved!")
    
    # Feature distributions by class
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(X.columns):
        axes[idx].hist([X[y==0][col], X[y==1][col]], bins=30, alpha=0.7, 
                      label=['Bad Wine (0)', 'Good Wine (1)'])
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature distributions saved!")


def split_and_scale_data(X, y):
    """Split data into train/test and scale features."""
    print("\n" + "="*70)
    print("SPLITTING AND SCALING DATA")
    print("="*70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training set class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test set class distribution: {y_test.value_counts().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


def train_classifiers(X_train, X_test, y_train, y_test):
    """Train multiple classification algorithms and compare performance."""
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS")
    print("="*70)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, trained_models


def evaluate_models(results, y_test):
    """Evaluate and compare all models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()],
        'F1-Score': [results[m]['f1'] for m in results.keys()],
        'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
        'CV Std': [results[m]['cv_std'] for m in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv('./model_comparison.csv', index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        comparison_df_sorted = comparison_df.sort_values(metric, ascending=True)
        ax.barh(comparison_df_sorted['Model'], comparison_df_sorted[metric], color='steelblue')
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(comparison_df_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nModel comparison plots saved!")
    
    # Best model
    best_model = comparison_df.iloc[0]['Model']
    print(f"\nBest performing model: {best_model}")
    print(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    
    return comparison_df, best_model


def analyze_feature_importance(trained_models, X_train, y_train, feature_names):
    """Analyze feature importance using multiple methods."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    importance_results = {}
    
    # 1. Random Forest Feature Importance
    if 'Random Forest' in trained_models:
        rf_model = trained_models['Random Forest']
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_results['Random Forest'] = rf_importance
        
        print("\nRandom Forest Feature Importance:")
        print(rf_importance.to_string(index=False))
    
    # 2. Gradient Boosting Feature Importance
    if 'Gradient Boosting' in trained_models:
        gb_model = trained_models['Gradient Boosting']
        gb_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': gb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_results['Gradient Boosting'] = gb_importance
        
        print("\nGradient Boosting Feature Importance:")
        print(gb_importance.to_string(index=False))
    
    # 3. Permutation Importance (use best tree-based model)
    print("\nCalculating Permutation Importance (this may take a moment)...")
    model_for_perm = trained_models.get('Random Forest') or trained_models.get('Gradient Boosting')
    if model_for_perm:
        perm_importance = permutation_importance(
            model_for_perm, X_train, y_train, n_repeats=10, random_state=42
        )
        perm_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        importance_results['Permutation'] = perm_imp_df
        
        print("\nPermutation Importance:")
        print(perm_imp_df.to_string(index=False))
    
    # Plot feature importance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Random Forest', 'Gradient Boosting', 'Permutation']
    for idx, method in enumerate(methods):
        if method in importance_results:
            imp_df = importance_results[method].head(10)
            axes[idx].barh(imp_df['Feature'], imp_df['Importance'], color='coral')
            axes[idx].set_xlabel('Importance', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{method} - Top 10 Features', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nFeature importance plots saved!")
    
    return importance_results


def plot_confusion_matrices(results, trained_models, y_test):
    """Plot confusion matrices for all models."""
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRICES")
    print("="*70)
    
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True)
        axes[idx].set_title(f'{name} - Accuracy: {result["accuracy"]:.3f}', 
                          fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('./confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrices saved!")


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning on best models."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    # Random Forest tuning
    print("\nTuning Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    
    print(f"Best RF parameters: {rf_grid.best_params_}")
    print(f"Best RF F1-score: {rf_grid.best_score_:.4f}")
    
    # Gradient Boosting tuning
    print("\nTuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    
    print(f"Best GB parameters: {gb_grid.best_params_}")
    print(f"Best GB F1-score: {gb_grid.best_score_:.4f}")
    
    return rf_grid.best_estimator_, gb_grid.best_estimator_


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("WINE QUALITY CLASSIFICATION")
    print("="*70)
    
    # Load data
    X, y, data = load_and_prepare_data()
    
    # Explore data
    explore_data(X, y)
    
    # Split and scale
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = split_and_scale_data(X, y)
    
    # Train classifiers
    results, trained_models = train_classifiers(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    comparison_df, best_model = evaluate_models(results, y_test)
    
    # Feature importance
    importance_results = analyze_feature_importance(
        trained_models, X_train, y_train, X.columns.tolist()
    )
    
    # Confusion matrices
    plot_confusion_matrices(results, trained_models, y_test)
    
    # Hyperparameter tuning
    best_rf, best_gb = hyperparameter_tuning(X_train, y_train)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBest Model: {best_model}")
    print(f"Test Accuracy: {results[best_model]['accuracy']:.4f}")
    print(f"Test F1-Score: {results[best_model]['f1']:.4f}")
    
    # Top 5 important features
    print("\nTop 5 Most Important Features:")
    if 'Random Forest' in importance_results:
        top_features = importance_results['Random Forest'].head(5)
        for idx, row in top_features.iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    print("\nAll outputs saved to ./")
    print("="*70)


if __name__ == "__main__":
    main()