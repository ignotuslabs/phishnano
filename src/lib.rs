//! # phishnano
//!
//! Lightweight offline phishing URL detection library with an embedded
//! Random Forest model. The model is compiled into the library at build
//! time via `include_bytes!`, enabling zero-configuration, fully local
//! inference with microsecond-level latency and no network requests.
//!
//! Designed for integration into password managers, browser extensions,
//! email security gateways, and embedded systems where privacy-preserving,
//! offline URL classification is required.
//!
//! ## Key Advantages
//!
//! - **Offline & privacy-preserving**: 100% local inference, zero network
//!   requests, no data leaves the host
//! - **Lightweight**: ~110 KB embedded model (bincode format)
//! - **Low latency**: ~20 microseconds per URL on commodity hardware
//! - **Zero configuration**: No runtime files, no API keys, no external
//!   services
//! - **Embedded-friendly**: Compact binary suitable for resource-constrained
//!   environments
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
//! ## Use Cases
//!
//! - **Password managers**: Warn users before autofilling credentials on
//!   suspicious login pages
//! - **Browser extensions**: Real-time URL classification during navigation
//! - **Email security gateways**: Scan links in incoming messages without
//!   forwarding URLs to cloud APIs
//! - **Security pipelines**: Batch URL classification in SOAR / SIEM
//!   workflows
//! - **Embedded systems**: On-device phishing detection in network
//!   appliances with limited connectivity
//!
//! ## Architecture
//!
//! - **Model**: Random Forest with 25 decision trees, max depth 7
//! - **Features**: 500 character n-gram hash features + 19 manual features
//! - **Model size**: ~110 KB (bincode format, embedded)
//! - **Inference latency**: ~20 microseconds per URL
//! - **Privacy**: 100% local inference, no network requests
//! - **Default threshold**: 0.45 (scores >= 0.45 are classified as phishing)
//!
//! ## Modules
//!
//! - [`model`]: Model serialization, deserialization, and loading
//! - [`extractor`]: Feature extraction from URL strings
//! - [`predictor`]: Decision tree traversal and scoring
//!
//! ## Core API
//!
//! - [`load_default_model()`]: Load the embedded default model (zero config)
//! - [`predict_url()`]: Predict phishing probability for a URL
//! - [`extract_features()`]: Extract the 519-dimensional feature vector
//! - [`Model`]: The Random Forest model struct
//! - [`Tree`]: A single decision tree in the forest
//! - [`load_model_from_path()`]: Load a model from a file path
//! - [`load_model_from_bytes()`]: Load a model from raw bytes
//! - [`convert_json_to_bincode()`]: Convert JSON model to bincode format

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
