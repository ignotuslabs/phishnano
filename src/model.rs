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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub n_features: usize,
    pub n_manual_features: usize,
    pub ngram_range: [usize; 2],
    pub trees: Vec<Tree>,
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
/// # Example
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
pub fn load_model_from_path(path: &str) -> Result<Model, anyhow::Error> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    load_model_from_bytes(&data)
}

/// Load a model from raw bytes, trying bincode first then JSON.
///
/// This is the core loading function used by both [`load_default_model`]
/// and [`load_model_from_path`]. It attempts bincode deserialization first
/// (since it is faster and more compact), then falls back to JSON if the
/// data does not appear to be valid bincode.
///
/// # Arguments
///
/// - `data`: Raw bytes of the model file
///
/// # Returns
///
/// - `Ok(Model)` if either format succeeds
/// - `Err` if both formats fail to deserialize
pub fn load_model_from_bytes(data: &[u8]) -> Result<Model, anyhow::Error> {
    if let Ok(model) = bincode::deserialize::<Model>(data) {
        return Ok(model);
    }
    let model: Model = serde_json::from_slice(data)?;
    Ok(model)
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
