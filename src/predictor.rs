//! Prediction module for phishing URL scoring.
//!
//! This module handles the inference logic: traversing decision trees
//! and aggregating their predictions into a final phishing probability
//! score. The prediction pipeline is:
//!
//! 1. Extract features from the URL via [`extract_features`]
//! 2. For each tree in the forest, traverse from root to leaf
//! 3. Sum the raw leaf values and apply `sigmoid(init_score + Σ raw_leaf)`
//!    to get the calibrated probability (LightGBM additive semantics)
//!
//! A score of 0.0 indicates "definitely normal", 1.0 indicates
//! "definitely phishing". The classification threshold is typically
//! set at 0.20, meaning scores >= 0.20 are classified as phishing.

use crate::extractor::extract_features;
use crate::model::{Model, Tree};

/// Maximum allowed tree traversal depth before bailing out.
///
/// This prevents infinite loops if a corrupted model file contains cyclic
/// node references. The embedded model has a maximum depth of 7, so 1000
/// provides ample headroom for any legitimate tree.
const MAX_TRAVERSAL_DEPTH: usize = 1000;

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
/// Returns `0.0` (neutral additive contribution) if the tree structure is
/// malformed or exceeds [`MAX_TRAVERSAL_DEPTH`] iterations (possible cycle).
///
/// # Examples
///
/// ```
/// use phishnano::predictor::predict_tree;
/// use phishnano::model::Tree;
///
/// let tree = Tree {
///     left: vec![-1],
///     right: vec![-1],
///     feature: vec![0],
///     threshold: vec![0.5],
///     value: vec![0.9],
/// };
/// let features = vec![0.3];
/// let score = predict_tree(&tree, &features);
/// assert_eq!(score, 0.9);
/// ```
pub fn predict_tree(tree: &Tree, features: &[f32]) -> f32 {
    let mut node = 0i32;

    for _ in 0..MAX_TRAVERSAL_DEPTH {
        let node_idx = node as usize;

        let left = match tree.left.get(node_idx) {
            Some(&l) => l,
            None => return 0.0,
        };

        if left == -1 {
            return tree.value.get(node_idx).copied().unwrap_or(0.5);
        }

        let feature_idx = tree.feature.get(node_idx).copied().unwrap_or(0);
        let threshold = tree.threshold.get(node_idx).copied().unwrap_or(0.0);

        let feature_val = features.get(feature_idx as usize).copied().unwrap_or(0.0);

        let next = if feature_val <= threshold {
            left
        } else {
            tree.right.get(node_idx).copied().unwrap_or(-1)
        };

        if next < 0 {
            return tree.value.get(node_idx).copied().unwrap_or(0.0);
        }

        node = next;
    }

    0.0
}

/// A single step in a decision tree traversal path.
///
/// Records which feature was evaluated, the split threshold, the actual
/// feature value for this URL, and which branch was taken.
#[derive(Debug, Clone)]
pub struct PathStep {
    /// Feature index used at this split node.
    pub feature_idx: i32,
    /// Split threshold at this node (`feature <= threshold` → left).
    pub threshold: f32,
    /// Actual feature value for the URL being predicted.
    pub feature_val: f32,
    /// `true` if `feature_val <= threshold` (left branch), `false` for right.
    pub went_left: bool,
}

/// Traverse a decision tree and return both the leaf prediction and the
/// full decision path.
///
/// This is the path-recording variant of [`predict_tree`], used by
/// [`predict_url_detailed`](crate::indicators::predict_url_detailed) to
/// extract risk indicators. The traversal logic is identical to
/// [`predict_tree`]; the only difference is that each internal node visited
/// is recorded as a [`PathStep`].
///
/// # Arguments
///
/// - `tree`: A single decision tree from the Random Forest
/// - `features`: The feature vector extracted from the URL
///
/// # Returns
///
/// A tuple of `(leaf_value, path)` where:
/// - `leaf_value` is the phishing probability at the reached leaf (0.0-1.0),
///   or 0.0 if the tree structure is malformed
/// - `path` is a vector of [`PathStep`] records, one per internal node visited
pub fn predict_tree_with_path(tree: &Tree, features: &[f32]) -> (f32, Vec<PathStep>) {
    let mut node = 0i32;
    let mut path = Vec::new();

    for _ in 0..MAX_TRAVERSAL_DEPTH {
        let node_idx = node as usize;

        let left = match tree.left.get(node_idx) {
            Some(&l) => l,
            None => return (0.0, path),
        };

        if left == -1 {
            return (tree.value.get(node_idx).copied().unwrap_or(0.5), path);
        }

        let feature_idx = tree.feature.get(node_idx).copied().unwrap_or(0);
        let threshold = tree.threshold.get(node_idx).copied().unwrap_or(0.0);

        let feature_val = features.get(feature_idx as usize).copied().unwrap_or(0.0);

        let went_left = feature_val <= threshold;
        let next = if went_left {
            left
        } else {
            tree.right.get(node_idx).copied().unwrap_or(-1)
        };

        path.push(PathStep {
            feature_idx,
            threshold,
            feature_val,
            went_left,
        });

        if next < 0 {
            return (tree.value.get(node_idx).copied().unwrap_or(0.0), path);
        }

        node = next;
    }

    (0.0, path)
}

/// Pure Random-Forest score (Stage 2 only, no Stage-1 rule layer).
///
/// This is the raw forest component used internally by the two-stage
/// [`crate::scoring::score_url`] scorer. Most callers should use
/// [`predict_url`], which applies the full two-stage pipeline
/// (deterministic whitelist / brand / high-risk-TLD layer + forest refinement).
///
/// # Panics
///
/// Panics if `model.trees` is empty, for the same safety reason as
/// [`predict_url`].
/// Logistic sigmoid, used to convert the LightGBM additive raw score into a
/// calibrated phishing probability in `[0, 1]`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn predict_forest(url: &str, model: &Model) -> f32 {
    if model.trees.is_empty() {
        panic!("predict_forest: model contains no trees — cannot produce a meaningful prediction");
    }

    let features = extract_features(
        url,
        model.n_features,
        model.n_manual_features,
        model.ngram_range,
    );
    // LightGBM is additive: each tree contributes a *raw* logit, and the final
    // probability is `sigmoid(init_score + Σ raw_leaf)`. The legacy sklearn
    // RandomForest export averaged leaf probabilities instead; the production
    // model is LightGBM, so we always apply the sigmoid here. `init_score`
    // defaults to 0.0 for any export that omitted it.
    let mut raw = 0.0f32;
    for tree in &model.trees {
        raw += predict_tree(tree, &features);
    }
    sigmoid(model.init_score + raw)
}

/// Predict the phishing probability of a URL using the full two-stage pipeline.
///
/// This is the main entry point for phishing detection. It applies the
/// deterministic **Stage-1** rule layer (whitelist fix / brand-impersonation
/// / high-risk TLD) and refines with the embedded **decision tree forest** (Stage 2).
/// See [`crate::scoring`] for the architecture and rationale.
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
/// - The default classification threshold is 0.20
///
/// # Panics
///
/// Panics if `model.trees` is empty. An empty model cannot produce a
/// meaningful prediction, and silently returning a default value (such as
/// 0.0 or NaN) would create a security risk where all URLs are classified
/// as safe. The embedded default model always contains 100 trees, so this
/// panic only occurs with custom models that were incorrectly constructed.
///
/// # Examples
///
/// ```no_run
/// use phishnano::{load_default_model, predict_url};
///
/// let model = load_default_model().unwrap();
/// let score = predict_url("http://suspicious-site.com/login", &model);
///
/// if score >= 0.20 {
///     println!("WARNING: Potential phishing site (score={:.4})", score);
/// } else {
///     println!("URL appears safe (score={:.4})", score);
/// }
/// ```
pub fn predict_url(url: &str, model: &Model) -> f32 {
    crate::scoring::score_url(url, model)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_predict_tree_with_split() {
        let tree = Tree {
            left: vec![1, -1, -1],
            right: vec![2, -1, -1],
            feature: vec![0, 0, 0],
            threshold: vec![1.0, 0.0, 0.0],
            value: vec![0.0, 0.2, 0.8],
        };
        let features_left = vec![0.5];
        assert_eq!(predict_tree(&tree, &features_left), 0.2);
        let features_right = vec![1.5];
        assert_eq!(predict_tree(&tree, &features_right), 0.8);
    }

    #[test]
    fn test_predict_tree_cycle_returns_neutral() {
        let tree = Tree {
            left: vec![1, 0],
            right: vec![-1, -1],
            feature: vec![0, 0],
            threshold: vec![1.0, 1.0],
            value: vec![0.0, 0.0],
        };
        let features = vec![0.5];
        let score = predict_tree(&tree, &features);
        assert_eq!(score, 0.0, "Cyclic tree should return neutral 0.0");
    }

    #[test]
    fn test_predict_tree_out_of_bounds_returns_neutral() {
        let tree = Tree {
            left: vec![5],
            right: vec![-1],
            feature: vec![0],
            threshold: vec![1.0],
            value: vec![0.0],
        };
        let features = vec![0.5];
        let score = predict_tree(&tree, &features);
        assert_eq!(score, 0.0, "Out-of-bounds node should return neutral 0.0");
    }

    #[test]
    #[should_panic(expected = "model contains no trees")]
    fn test_predict_url_empty_trees_panics() {
        let model = Model {
            n_features: 10,
            n_manual_features: 5,
            ngram_range: [2, 3],
            init_score: 0.0,
            trees: vec![],
        };
        let _ = predict_url("http://example.com", &model);
    }

    #[test]
    fn test_predict_tree_with_path_single_node() {
        let tree = Tree {
            left: vec![-1],
            right: vec![-1],
            feature: vec![0],
            threshold: vec![0.0],
            value: vec![0.5],
        };
        let features = vec![1.0];
        let (score, path) = predict_tree_with_path(&tree, &features);
        assert_eq!(score, 0.5);
        assert!(path.is_empty(), "Single leaf node should have empty path");
    }

    #[test]
    fn test_predict_tree_with_path_left_branch() {
        let tree = Tree {
            left: vec![1, -1, -1],
            right: vec![2, -1, -1],
            feature: vec![0, 0, 0],
            threshold: vec![1.0, 0.0, 0.0],
            value: vec![0.0, 0.2, 0.8],
        };
        let features = vec![0.5]; // 0.5 <= 1.0 → left
        let (score, path) = predict_tree_with_path(&tree, &features);
        assert_eq!(score, 0.2);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].feature_idx, 0);
        assert_eq!(path[0].feature_val, 0.5);
        assert_eq!(path[0].threshold, 1.0);
        assert!(path[0].went_left);
    }

    #[test]
    fn test_predict_tree_with_path_right_branch() {
        let tree = Tree {
            left: vec![1, -1, -1],
            right: vec![2, -1, -1],
            feature: vec![0, 0, 0],
            threshold: vec![1.0, 0.0, 0.0],
            value: vec![0.0, 0.2, 0.8],
        };
        let features = vec![1.5]; // 1.5 > 1.0 → right
        let (score, path) = predict_tree_with_path(&tree, &features);
        assert_eq!(score, 0.8);
        assert_eq!(path.len(), 1);
        assert!(!path[0].went_left);
    }

    #[test]
    fn test_predict_tree_with_path_multi_depth() {
        // Root: feature 0 <= 0.5 → left (node 1)
        // Node 1: feature 1 <= 0.5 → left (node 3, leaf)
        let tree = Tree {
            left: vec![1, 3, -1, -1],
            right: vec![2, -1, -1, -1],
            feature: vec![0, 1, 0, 0],
            threshold: vec![0.5, 0.5, 0.0, 0.0],
            value: vec![0.0, 0.0, 0.8, 0.3],
        };
        let features = vec![0.3, 0.4];
        let (score, path) = predict_tree_with_path(&tree, &features);
        assert_eq!(score, 0.3);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].feature_idx, 0);
        assert!(path[0].went_left);
        assert_eq!(path[1].feature_idx, 1);
        assert!(path[1].went_left);
    }
}
