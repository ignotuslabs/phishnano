//! Prediction module for phishing URL scoring.
//!
//! This module handles the inference logic: traversing decision trees
//! and aggregating their predictions into a final phishing probability
//! score. The prediction pipeline is:
//!
//! 1. Extract features from the URL via [`extract_features`]
//! 2. For each tree in the forest, traverse from root to leaf
//! 3. Average all tree predictions to get the final score
//!
//! A score of 0.0 indicates "definitely normal", 1.0 indicates
//! "definitely phishing". The classification threshold is typically
//! set at 0.45, meaning scores >= 0.45 are classified as phishing.

use crate::extractor::extract_features;
use crate::model::{Model, Tree};

/// Traverse a single decision tree and return the leaf node's prediction value.
///
/// The tree is represented as a flat array structure (see [`Tree`]). Starting
/// from the root node (index 0), the function follows the split rules:
///
/// - If `feature[node] <= threshold[node]`, go to `left[node]`
/// - Otherwise, go to `right[node]`
///
/// The traversal continues until a leaf node is reached (marked by
/// `left[node] == -1`), at which point `value[node]` is returned.
///
/// # Arguments
///
/// - `tree`: A single decision tree from the Random Forest
/// - `features`: The feature vector extracted from the URL
///
/// # Returns
///
/// The phishing probability (0.0 to 1.0) stored at the leaf node.
///
/// # Panics
///
/// This function will panic if the tree structure is malformed (e.g.,
/// contains cycles or invalid child indices). The training pipeline
/// guarantees well-formed trees.
pub fn predict_tree(tree: &Tree, features: &[f32]) -> f32 {
    let mut node = 0i32;
    loop {
        // A left child of -1 indicates a leaf node.
        if tree.left[node as usize] == -1 {
            return tree.value[node as usize];
        }
        // Internal node: compare the feature value against the threshold.
        let feature_idx = tree.feature[node as usize];
        let threshold = tree.threshold[node as usize];
        if features[feature_idx as usize] <= threshold {
            node = tree.left[node as usize];
        } else {
            node = tree.right[node as usize];
        }
    }
}

/// Predict the phishing probability of a URL using the full Random Forest model.
///
/// This is the main entry point for phishing detection. It extracts features
/// from the URL, runs them through every tree in the forest, and returns the
/// average prediction (phishing probability).
///
/// # Arguments
///
/// - `url`: The raw URL string to classify
/// - `model`: The loaded Random Forest model
///
/// # Returns
///
/// A floating-point score between 0.0 and 1.0:
/// - Scores close to 0.0 → likely a legitimate URL
/// - Scores close to 1.0 → likely a phishing URL
/// - The default classification threshold is 0.45
///
/// # Example
///
/// ```no_run
/// use phishnano::{load_default_model, predict_url};
///
/// let model = load_default_model().unwrap();
/// let score = predict_url("http://suspicious-site.com/login", &model);
///
/// if score >= 0.45 {
///     println!("WARNING: Potential phishing site (score={:.4})", score);
/// } else {
///     println!("URL appears safe (score={:.4})", score);
/// }
/// ```
pub fn predict_url(url: &str, model: &Model) -> f32 {
    let features = extract_features(url, model.n_features, model.n_manual_features);
    let mut sum = 0.0;
    for tree in &model.trees {
        sum += predict_tree(tree, &features);
    }
    sum / model.trees.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a single-node tree (leaf only) returns its stored value
    /// regardless of the input features.
    #[test]
    fn test_predict_tree_single_node() {
        let tree = Tree {
            left: vec![-1],
            right: vec![-1],
            feature: vec![0],
            threshold: vec![0.0],
            value: vec![0.5],
        };
        let features = vec![1.0];
        assert_eq!(predict_tree(&tree, &features), 0.5);
    }

    /// Verify that tree traversal follows the correct branch based on
    /// the feature/threshold comparison. A feature value below the
    /// threshold should follow the left child, and above should follow
    /// the right child.
    #[test]
    fn test_predict_tree_with_split() {
        let tree = Tree {
            left: vec![1, -1, -1],
            right: vec![2, -1, -1],
            feature: vec![0, 0, 0],
            threshold: vec![1.0, 0.0, 0.0],
            value: vec![0.0, 0.2, 0.8],
        };
        // Feature value 0.5 <= threshold 1.0 → go left → node 1 → leaf value 0.2
        let features_left = vec![0.5];
        assert_eq!(predict_tree(&tree, &features_left), 0.2);
        // Feature value 1.5 > threshold 1.0 → go right → node 2 → leaf value 0.8
        let features_right = vec![1.5];
        assert_eq!(predict_tree(&tree, &features_right), 0.8);
    }
}
