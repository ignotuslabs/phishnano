//! Phishing URL scoring with the embedded decision tree forest.
//!
//! Production scorer for phishnano. Delegates entirely to the embedded
//! LightGBM decision tree forest ([`predict_forest`]) — no external rule
//! layer is applied at inference time.
//!
//! The forest produces a calibrated phishing probability in `[0, 1]` via
//! `sigmoid(init_score + Σ raw_leaf)`. The default classification threshold
//! is 0.20 (scores >= 0.20 are classified as phishing).

use crate::model::Model;
use crate::predictor::predict_forest;

/// Phishing URL score for a URL.
///
/// Returns the forest's calibrated phishing probability in `[0, 1]`.
/// The default classification threshold is 0.20.
pub fn score_url(url: &str, model: &Model) -> f32 {
    predict_forest(url, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_default_model;

    /// Default classification threshold (matches the documented deployment
    /// value; scores >= this are "Phishing").
    const THRESHOLD: f32 = 0.20;

    #[test]
    fn test_score_url_normal_and_phishing() {
        let model = load_default_model().expect("Failed to load model");
        // Normal URL -> below threshold.
        let n = score_url("https://www.google.com", &model);
        assert!(n < THRESHOLD);
        // Brand typosquat -> flagged as phishing by the forest.
        let p = score_url("http://paypa1.com/login", &model);
        assert!(p >= THRESHOLD);
        // High-risk TLD typosquat -> flagged as phishing by the forest.
        let p2 = score_url("http://a1b2c3.tk/login", &model);
        assert!(p2 >= THRESHOLD);
    }
}
