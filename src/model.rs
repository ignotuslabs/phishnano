//! Model serialization and loading module.
//!
//! This module handles the serialization, deserialization, and loading of
//! the Random Forest model used for phishing URL detection. The model is
//! stored in two formats:
//!
//! - **JSON** (`model_data.json`): Human-readable format used during
//!   development and debugging.
//! - **Bincode** (`model_data.bincode`): Compact binary format embedded
//!   into the library via `include_bytes!` for zero-configuration usage
//!   in production.
//!
//! The bincode format is approximately 75% smaller than JSON, reducing
//! the embedded model from ~419 KB to ~103 KB.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Embedded default model in bincode format.
///
/// This constant is compiled into the library binary at build time,
/// allowing users to call [`load_default_model`] without specifying
/// a file path. The model file is located at `resources/model_data.bincode`
/// relative to the project root.
const DEFAULT_MODEL_BYTES: &[u8] = include_bytes!("../resources/model_data.bincode");

/// A single decision tree in the Random Forest.
///
/// Each field is a flat array representation of the tree structure, where
/// index `i` corresponds to node `i`. The tree is traversed starting from
/// node 0, following left/right child pointers until a leaf node (marked
/// by `left == -1`) is reached.
///
/// # Fields
///
/// - `left`: Left child indices (-1 indicates a leaf node)
/// - `right`: Right child indices (-1 indicates a leaf node)
/// - `feature`: Feature index used for splitting at each internal node
/// - `threshold`: Threshold value for the split (feature <= threshold → left)
/// - `value`: Prediction value stored at leaf nodes (phishing probability)
///
/// # Examples
///
/// ```
/// use phishnano::model::Tree;
///
/// let tree = Tree {
///     left: vec![-1],
///     right: vec![-1],
///     feature: vec![0],
///     threshold: vec![0.5],
///     value: vec![0.9],
/// };
/// assert_eq!(tree.value[0], 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tree {
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
}

/// The complete Random Forest model for phishing URL detection.
///
/// # Fields
///
/// - `n_features`: Number of n-gram hash features (typically 500)
/// - `n_manual_features`: Number of manual engineered features (typically 19)
/// - `ngram_range`: Character n-gram range `[min, max]` (typically `[2, 3]`)
/// - `trees`: Collection of decision trees in the forest
///
/// # Feature Layout
///
/// The feature vector has `n_features + n_manual_features` dimensions:
/// - Indices `[0, n_features)`: Character n-gram hash counts
/// - Indices `[n_features, n_features + n_manual_features)`: Manual features
///
/// # Examples
///
/// ```
/// use phishnano::model::{Model, Tree};
///
/// let model = Model {
///     n_features: 500,
///     n_manual_features: 19,
///     ngram_range: [2, 3],
///     trees: vec![Tree {
///         left: vec![-1],
///         right: vec![-1],
///         feature: vec![0],
///         threshold: vec![0.5],
///         value: vec![0.8],
///     }],
/// };
/// assert_eq!(model.trees.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub n_features: usize,
    pub n_manual_features: usize,
    pub ngram_range: [usize; 2],
    pub trees: Vec<Tree>,
}

impl Model {
    /// Extract the feature vector for a URL using this model's configuration.
    ///
    /// This is the recommended way to extract features, as it uses the
    /// model's own `n_features`, `n_manual_features`, and `ngram_range`
    /// values, eliminating the risk of parameter mismatch.
    ///
    /// # Arguments
    ///
    /// - `url`: The raw URL string to analyze
    ///
    /// # Returns
    ///
    /// A feature vector of `n_features + n_manual_features` dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use phishnano::model::{Model, Tree};
    ///
    /// let model = Model {
    ///     n_features: 10,
    ///     n_manual_features: 5,
    ///     ngram_range: [2, 3],
    ///     trees: vec![],
    /// };
    /// let features = model.extract_features("https://example.com");
    /// assert_eq!(features.len(), 15);
    /// ```
    pub fn extract_features(&self, url: &str) -> Vec<f32> {
        crate::extractor::extract_features(
            url,
            self.n_features,
            self.n_manual_features,
            self.ngram_range,
        )
    }
}

/// Load the default embedded model (bincode format, zero configuration).
///
/// This function loads the model that was compiled into the library at
/// build time via `include_bytes!`. It requires no file path and is the
/// recommended way for end users to load the model.
///
/// # Returns
///
/// - `Ok(Model)` on successful deserialization
/// - `Err` if the embedded model data is corrupted
///
/// # Errors
///
/// Returns an error if the embedded bincode data is corrupted or cannot
/// be deserialized. This should never happen under normal circumstances,
/// as the embedded model is validated at build time.
///
/// # Examples
///
/// ```no_run
/// use phishnano::load_default_model;
/// use phishnano::predict_url;
///
/// let model = load_default_model().expect("Failed to load model");
/// let score = predict_url("http://suspicious.com", &model);
/// ```
pub fn load_default_model() -> Result<Model, anyhow::Error> {
    load_model_from_bytes(DEFAULT_MODEL_BYTES)
}

/// Load a model from a file path, auto-detecting bincode or JSON format.
///
/// The function reads the entire file into memory and attempts bincode
/// deserialization first. If that fails, it falls back to JSON. This
/// allows users to load either format without specifying the type.
///
/// # Arguments
///
/// - `path`: Path to the model file (`.json` or `.bincode`)
///
/// # Returns
///
/// - `Ok(Model)` on successful load
/// - `Err` if the file cannot be read or deserialized
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be read (IO error)
/// - The file content is neither valid bincode nor valid JSON
///   (deserialization error)
///
/// # Examples
///
/// ```no_run
/// use phishnano::load_model_from_path;
///
/// // Load a bincode or JSON model from a file
/// let model = load_model_from_path("my_model.bincode")
///     .expect("Failed to load model");
/// ```
pub fn load_model_from_path(path: &str) -> Result<Model, anyhow::Error> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    load_model_from_bytes(&data)
}

/// Load a model from raw bytes, auto-detecting bincode or JSON format.
///
/// Format detection is based on the first and last non-whitespace bytes:
/// - JSON starts with `{` (0x7b) and ends with `}` (0x7d)
/// - Bincode is any other binary format
///
/// This explicit detection avoids the risk of bincode silently accepting
/// JSON or corrupted data as a garbage model (bincode has no magic number
/// or checksum to reject invalid input).
///
/// # Arguments
///
/// - `data`: Raw bytes of the model file
///
/// # Returns
///
/// - `Ok(Model)` if the data is valid JSON or bincode
/// - `Err` if deserialization fails for the detected format
///
/// # Errors
///
/// Returns an error if:
/// - The data is detected as JSON but `serde_json::from_slice` fails
/// - The data is detected as bincode but `bincode::deserialize` fails
///
/// # Examples
///
/// ```
/// use phishnano::load_model_from_bytes;
///
/// // Load a model from a JSON byte slice
/// let json = br#"{"n_features":10,"n_manual_features":5,"ngram_range":[2,3],"trees":[]}"#;
/// let model = load_model_from_bytes(json).expect("Failed to load");
/// assert_eq!(model.n_features, 10);
/// ```
pub fn load_model_from_bytes(data: &[u8]) -> Result<Model, anyhow::Error> {
    let first = data.iter().find(|&&b| !b.is_ascii_whitespace());
    let last = data.iter().rfind(|&&b| !b.is_ascii_whitespace());

    if first == Some(&b'{') && last == Some(&b'}') {
        let model: Model = serde_json::from_slice(data)?;
        Ok(model)
    } else {
        let model: Model = bincode::deserialize(data)?;
        Ok(model)
    }
}

/// Convert a JSON model file to bincode format and write to output path.
///
/// This function is used after training to produce the compact bincode
/// model that gets embedded into the library. The typical workflow is:
///
/// 1. Python `train.py` exports the model as JSON
/// 2. CLI `--convert` calls this function to produce bincode
/// 3. Bincode file is placed in `resources/model_data.bincode`
/// 4. Library is rebuilt to embed the new bincode via `include_bytes!`
///
/// # Arguments
///
/// - `json_path`: Path to the input JSON model file
/// - `bincode_path`: Path for the output bincode model file
///
/// # Returns
///
/// - `Ok(u64)`: Size of the bincode output in bytes
/// - `Err` if reading JSON or writing bincode fails
///
/// # Errors
///
/// Returns an error if:
/// - The input JSON file cannot be read or parsed
/// - The output bincode file cannot be written
///
/// # Examples
///
/// ```no_run
/// use phishnano::convert_json_to_bincode;
///
/// let size = convert_json_to_bincode("model_data.json", "model_data.bincode")
///     .expect("Conversion failed");
/// println!("Bincode size: {} bytes", size);
/// ```
pub fn convert_json_to_bincode<P: AsRef<Path>>(
    json_path: P,
    bincode_path: P,
) -> Result<u64, anyhow::Error> {
    let mut file = File::open(json_path.as_ref())?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let model: Model = serde_json::from_slice(&data)?;
    let encoded = bincode::serialize(&model)?;

    let mut out = File::create(bincode_path.as_ref())?;
    out.write_all(&encoded)?;
    Ok(encoded.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a Model can be serialized to bincode and back without
    /// any data loss (round-trip test).
    #[test]
    fn test_model_serialization_roundtrip() {
        let model = Model {
            n_features: 10,
            n_manual_features: 5,
            ngram_range: [2, 3],
            trees: vec![Tree {
                left: vec![-1],
                right: vec![-1],
                feature: vec![0],
                threshold: vec![0.5],
                value: vec![0.8],
            }],
        };

        let encoded = bincode::serialize(&model).unwrap();
        let decoded: Model = bincode::deserialize(&encoded).unwrap();
        assert_eq!(model, decoded);
    }

    /// Verify that bincode produces a smaller output than JSON for the
    /// same model, confirming the size advantage of the binary format.
    #[test]
    fn test_bincode_smaller_than_json() {
        let model = Model {
            n_features: 500,
            n_manual_features: 19,
            ngram_range: [2, 3],
            trees: vec![Tree {
                left: vec![1, -1, -1],
                right: vec![2, -1, -1],
                feature: vec![0, 0, 0],
                threshold: vec![1.0, 0.0, 0.0],
                value: vec![0.0, 0.2, 0.8],
            }],
        };

        let bincode_size = bincode::serialize(&model).unwrap().len();
        let json_size = serde_json::to_vec(&model).unwrap().len();
        assert!(
            bincode_size < json_size,
            "bincode ({}) should be smaller than JSON ({})",
            bincode_size,
            json_size
        );
    }
}
