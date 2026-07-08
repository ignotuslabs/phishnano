//! # phishnano
//!
//! Lightweight offline phishing URL detection library with an embedded
//! decision tree forest model (LightGBM-trained). The model is compiled into the library at build
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
//! - **Lightweight**: ~123 KB embedded model (bincode format)
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
//! if score >= 0.20 {
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
//! - **Model**: LightGBM Random Forest, 100 decision trees, max depth 7
//!   (additive `sigmoid(init_score + Σ raw_leaf)` scoring)
//! - **Features**: 500 character n-gram hash features + 39 manual features
//!   (21 hand-crafted + 18 structural)
//! - **Model size**: ~120 KB (bincode format, embedded)
//! - **Scoring**: Whitelist safety net (zero ML volume) + decision tree forest
//! - **Inference latency**: ~20 microseconds per URL
//! - **Privacy**: 100% local inference, no network requests
//! - **Default threshold**: 0.20 (scores >= 0.20 are classified as phishing)
//!
//! ## Modules
//!
//! - [`model`]: Model serialization, deserialization, and loading
//! - [`extractor`]: Feature extraction from URL strings
//! - [`predictor`]: Decision tree traversal and scoring
//! - [`indicators`]: Detailed risk indicator extraction
//! - [`scoring`]: Whitelist-backed scorer (whitelist safety net + forest)
//!
//! ## Core API
//!
//! - [`load_default_model()`]: Load the embedded default model (zero config)
//! - [`predict_url()`]: Predict phishing probability for a URL
//! - [`predict_url_detailed()`]: Predict with risk indicators (explains _why_)
//! - [`extract_features()`]: Extract the 539-dimensional feature vector
//! - [`Model`]: The decision tree forest model struct
//! - [`Tree`]: A single decision tree in the forest
//! - [`load_model_from_path()`]: Load a model from a file path
//! - [`load_model_from_bytes()`]: Load a model from raw bytes
//! - [`convert_json_to_bincode()`]: Convert JSON model to bincode format

pub mod extractor;
pub mod indicators;
pub mod model;
pub mod predictor;
pub mod scoring;

// Re-export the primary API for user convenience.
pub use extractor::extract_features;
pub use indicators::{
    predict_url_detailed, Indicator, IndicatorCategory, IndicatorSource, Prediction,
};
pub use model::{
    convert_json_to_bincode, load_default_model, load_model_from_bytes, load_model_from_path,
    Model, Tree,
};
pub use predictor::predict_url;

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: verify that feature extraction produces a vector
    /// of the expected length (500 n-gram + 39 manual = 539 features).
    #[test]
    fn test_integration() {
        let features = extract_features("example.com", 500, 39, [2, 3]);
        assert_eq!(features.len(), 539);
    }

    /// Verify that the embedded default model loads successfully and has
    /// the expected configuration (500 n-gram features, 39 manual features).
    /// Rust extracts 40 manual features (21 engineered + 19 structural); the
    /// embedded legacy model (`n_manual_features = 39`) uses the first 39 and
    /// ignores the trailing structural feature.
    #[test]
    fn test_embedded_model_loads() {
        let model = load_default_model().expect("Failed to load embedded model");
        assert_eq!(model.n_features, 500);
        assert_eq!(model.n_manual_features, 39);
        assert!(!model.trees.is_empty(), "Model should have trees");
    }

    /// Verify that the embedded model correctly distinguishes between
    /// a known legitimate URL and a known phishing URL.
    #[test]
    fn test_embedded_model_prediction() {
        let model = load_default_model().expect("Failed to load embedded model");
        let normal_score = predict_url("https://www.google.com", &model);
        let phishing_score = predict_url(
            "nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com",
            &model,
        );
        assert!(
            normal_score < 0.20,
            "Normal URL should score below threshold: {}",
            normal_score
        );
        assert!(
            phishing_score >= 0.20,
            "Phishing URL should score above threshold: {}",
            phishing_score
        );
    }

    /// Cross-validate Rust feature extraction against Python reference data.
    /// `resources/test_features.json` is embedded at compile time via
    /// `include_str!` and its features are compared against the Rust-extracted
    /// features for each URL. This ensures feature consistency between
    /// training (Python) and inference (Rust). The fixture is required: if it
    /// fails to parse, the test panics rather than silently passing.
    #[test]
    fn test_feature_extraction_consistency() {
        let content = include_str!("../resources/test_features.json");
        let data: serde_json::Value = serde_json::from_str(content).expect("Failed to parse JSON");

        for (url, value) in data.as_object().unwrap() {
            let cleaned = value["cleaned"].as_str().unwrap();
            let py_features: Vec<f32> = serde_json::from_value(value["features"].clone()).unwrap();
            let rust_features = extract_features(url, 500, 0, [2, 3]);

            // Guard against the Python reference silently dropping a dimension:
            // `zip` would otherwise truncate to the shorter vector and hide the gap.
            assert_eq!(
                py_features.len(),
                500,
                "Python reference must export 500 features for URL: {}",
                url
            );
            assert_eq!(
                rust_features.len(),
                500,
                "Rust extract_features must return 500 features for URL: {}",
                url
            );

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
