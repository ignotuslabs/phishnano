//! # phishnano
//!
//! A lightweight, embedded phishing URL detection library powered by a
//! Random Forest model. The model is compiled into the library at build
//! time, enabling zero-configuration usage with microsecond-level inference.
//!
//! ## Quick Start
//!
//! ```no_run
//! use phishnano::{load_default_model, predict_url};
//!
//! let model = load_default_model().expect("Failed to load model");
//! let score = predict_url("http://suspicious-site.com/login", &model);
//!
//! if score >= 0.45 {
//!     println!("Phishing detected (score={:.4})", score);
//! } else {
//!     println!("URL is safe (score={:.4})", score);
//! }
//! ```
//!
//! ## Architecture
//!
//! - **Model**: Random Forest with 25 decision trees, max depth 7
//! - **Features**: 500 character n-gram hash features + 19 manual features
//! - **Model size**: ~100 KB (bincode format, embedded)
//! - **Inference latency**: ~20 microseconds per URL
//! - **Privacy**: 100% local inference, no network requests
//!
//! ## Modules
//!
//! - [`model`]: Model serialization, deserialization, and loading
//! - [`extractor`]: Feature extraction from URL strings
//! - [`predictor`]: Decision tree traversal and scoring

pub mod extractor;
pub mod model;
pub mod predictor;

// Re-export the primary API for user convenience.
pub use extractor::extract_features;
pub use model::{
    convert_json_to_bincode, load_default_model, load_model_from_bytes, load_model_from_path,
    Model, Tree,
};
pub use predictor::predict_url;

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: verify that feature extraction produces a vector
    /// of the expected length (500 n-gram + 19 manual = 519 features).
    #[test]
    fn test_integration() {
        let features = extract_features("example.com", 500, 19);
        assert_eq!(features.len(), 519);

        let model = Model {
            n_features: 500,
            n_manual_features: 19,
            ngram_range: [2, 3],
            trees: vec![],
        };
        let score = predict_url("example.com", &model);
        assert!(score.is_nan() || score >= 0.0);
    }

    /// Verify that the embedded default model loads successfully and has
    /// the expected configuration (500 n-gram features, 19 manual features).
    #[test]
    fn test_embedded_model_loads() {
        let model = load_default_model().expect("Failed to load embedded model");
        assert_eq!(model.n_features, 500);
        assert_eq!(model.n_manual_features, 19);
        assert!(!model.trees.is_empty(), "Model should have trees");
    }

    /// Verify that the embedded model correctly distinguishes between
    /// a known legitimate URL and a known phishing URL.
    #[test]
    fn test_embedded_model_prediction() {
        let model = load_default_model().expect("Failed to load embedded model");
        let normal_score = predict_url("http://example.com", &model);
        let phishing_score = predict_url(
            "nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com",
            &model,
        );
        assert!(
            normal_score < 0.45,
            "Normal URL should score below threshold: {}",
            normal_score
        );
        assert!(
            phishing_score >= 0.45,
            "Phishing URL should score above threshold: {}",
            phishing_score
        );
    }

    /// Cross-validate Rust feature extraction against Python reference data.
    /// This test loads `resources/test_features.json` (if present) and
    /// compares the Rust-extracted features against the Python-extracted
    /// features for each URL. This ensures feature consistency between
    /// training (Python) and inference (Rust).
    #[test]
    fn test_feature_extraction_consistency() {
        let json_path = "../resources/test_features.json";
        if !std::fs::exists(json_path).unwrap_or(false) {
            println!("test_features.json not found, skipping test");
            return;
        }
        let content =
            std::fs::read_to_string(json_path).expect("Failed to read test_features.json");
        let data: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse JSON");

        for (url, value) in data.as_object().unwrap() {
            let cleaned = value["cleaned"].as_str().unwrap();
            let py_features: Vec<f32> = serde_json::from_value(value["features"].clone()).unwrap();
            let rust_features = extract_features(url, 500, 0);

            let py_nonzero: Vec<(usize, f32)> = py_features
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.001)
                .map(|(i, &v)| (i, v))
                .collect();
            let rust_nonzero: Vec<(usize, f32)> = rust_features
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.001)
                .map(|(i, &v)| (i, v))
                .collect();

            println!("URL: {}", url);
            println!("Cleaned: {}", cleaned);
            println!("Python nonzero ({}): {:?}", py_nonzero.len(), py_nonzero);
            println!("Rust nonzero ({}): {:?}", rust_nonzero.len(), rust_nonzero);

            let mut mismatches = 0;
            for (py, rust) in py_features.iter().zip(rust_features.iter()) {
                if (py - rust).abs() > 0.001 {
                    mismatches += 1;
                }
            }

            if mismatches > 0 {
                println!("Total mismatches: {}/500", mismatches);
            }

            assert_eq!(
                mismatches, 0,
                "Feature extraction mismatch for URL: {}",
                url
            );
        }
    }
}
