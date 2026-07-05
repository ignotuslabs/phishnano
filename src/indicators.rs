//! Risk indicator extraction for detailed phishing URL analysis.
//!
//! This module provides [`predict_url_detailed`], which returns both the
//! phishing probability score and a list of human-readable risk indicators
//! explaining _why_ the URL was classified as phishing (or normal).
//!
//! Indicators come from two sources:
//!
//! - **Model decision indicators** (方案 B): Derived by tracing the actual
//!   decision paths through all trees in the Random Forest. Features that
//!   were used by many trees are reported with their vote count (e.g.,
//!   "22/25 trees").
//! - **Heuristic indicators** (方案 A): Derived by checking the 19 manual
//!   features for abnormal values (e.g., IP address in domain, excessive
//!   subdomains, sensitive keywords). These supplement the model indicators
//!   with interpretable signals.
//!
//! # Example
//!
//! ```no_run
//! use phishnano::{load_default_model, predict_url_detailed};
//!
//! let model = load_default_model().expect("Failed to load model");
//! let result = predict_url_detailed("http://suspicious-site.com/login", &model);
//!
//! println!("Score: {:.4}", result.score);
//! for ind in &result.indicators {
//!     println!("  - {}", ind.description);
//! }
//! ```

use crate::extractor::{extract_manual_features, ngrams_for_bucket};
use crate::model::Model;
use crate::predictor::{predict_tree_with_path, predict_url};
use std::collections::HashMap;

/// Maximum number of indicators to return.
const MAX_INDICATORS: usize = 5;

/// Minimum number of trees that must use a feature for it to be reported
/// as a model decision indicator.
const MIN_TREE_VOTES: usize = 3;

/// The result of a detailed phishing URL prediction.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Phishing probability score (0.0 = definitely normal, 1.0 = definitely phishing).
    pub score: f32,
    /// Human-readable risk indicators explaining the score (max 5).
    pub indicators: Vec<Indicator>,
}

/// A single risk indicator.
#[derive(Debug, Clone)]
pub struct Indicator {
    /// Category of the risk factor.
    pub category: IndicatorCategory,
    /// Human-readable description of the risk.
    pub description: String,
    /// Contribution weight (0.0-1.0). Model indicators use `tree_count/total_trees`;
    /// heuristic indicators use a fixed 0.5.
    pub weight: f32,
    /// Whether this indicator comes from model decision tracing or a heuristic rule.
    pub source: IndicatorSource,
}

/// Category of a risk indicator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndicatorCategory {
    /// Related to the domain (IP usage, TLD, subdomain structure).
    Domain,
    /// Related to the URL path (sensitive keywords, path length).
    Path,
    /// Related to overall URL structure (length, special characters, digit ratio).
    Structure,
    /// Related to character n-gram patterns detected by the model.
    NGram,
}

/// The source of an indicator.
#[derive(Debug, Clone)]
pub enum IndicatorSource {
    /// Derived from model decision path tracing.
    /// `tree_count` = how many trees used this feature; `total_trees` = forest size.
    Model {
        tree_count: usize,
        total_trees: usize,
    },
    /// Derived from a heuristic rule on manual features.
    Heuristic,
}

/// Predict the phishing probability of a URL with detailed risk indicators.
///
/// This is the detailed variant of [`predict_url`](crate::predict_url). It
/// returns the same score plus a list of 0-5 risk indicators explaining the
/// classification. The score is computed identically to `predict_url`;
/// the indicators are additional context.
///
/// # Arguments
///
/// - `url`: The raw URL string to classify
/// - `model`: The loaded Random Forest model
///
/// # Returns
///
/// A [`Prediction`] containing the score and risk indicators.
///
/// # Panics
///
/// Panics if `model.trees` is empty (same as [`predict_url`](crate::predict_url)).
///
/// # Example
///
/// ```no_run
/// use phishnano::{load_default_model, predict_url_detailed};
///
/// let model = load_default_model().unwrap();
/// let result = predict_url_detailed("http://suspicious.com/login", &model);
/// println!("Score: {:.4}", result.score);
/// for ind in &result.indicators {
///     println!("  {}", ind.description);
/// }
/// ```
pub fn predict_url_detailed(url: &str, model: &Model) -> Prediction {
    let score = predict_url(url, model);
    let total_trees = model.trees.len();

    let features = model.extract_features(url);
    let manual_features = extract_manual_features(url);

    // --- 方案 B: Trace model decision paths ---
    let mut feature_votes: HashMap<i32, usize> = HashMap::new();
    for tree in &model.trees {
        let (_, path) = predict_tree_with_path(tree, &features);
        for step in &path {
            *feature_votes.entry(step.feature_idx).or_insert(0) += 1;
        }
    }

    // Sort features by vote count descending
    let mut ranked_features: Vec<(i32, usize)> = feature_votes
        .into_iter()
        .filter(|&(_, count)| count >= MIN_TREE_VOTES)
        .collect();
    ranked_features.sort_by_key(|b| std::cmp::Reverse(b.1));

    let mut indicators: Vec<Indicator> = Vec::new();
    let mut covered_manual: Vec<usize> = Vec::new();

    // Priority 1: Model manual-feature indicators (most interpretable model signals)
    for (feature_idx, tree_count) in &ranked_features {
        if indicators.len() >= MAX_INDICATORS {
            break;
        }
        let feat_idx = *feature_idx;
        if (feat_idx as usize) >= model.n_features {
            let manual_idx = feat_idx as usize - model.n_features;
            if manual_idx >= manual_features.len() {
                continue;
            }
            let val = manual_features[manual_idx];
            if let Some((category, desc)) = manual_feature_indicator(manual_idx, val, url) {
                covered_manual.push(manual_idx);
                indicators.push(Indicator {
                    category,
                    description: desc,
                    weight: *tree_count as f32 / total_trees as f32,
                    source: IndicatorSource::Model {
                        tree_count: *tree_count,
                        total_trees,
                    },
                });
            }
        }
    }

    // Priority 2: Heuristic indicators for abnormal manual features not already covered
    for (idx, &val) in manual_features.iter().enumerate() {
        if indicators.len() >= MAX_INDICATORS {
            break;
        }
        if covered_manual.contains(&idx) {
            continue;
        }
        if let Some((category, desc)) = manual_feature_indicator(idx, val, url) {
            covered_manual.push(idx);
            indicators.push(Indicator {
                category,
                description: desc,
                weight: 0.5,
                source: IndicatorSource::Heuristic,
            });
        }
    }

    // Priority 3: High-risk TLD check (not in the 19 manual features)
    if indicators.len() < MAX_INDICATORS {
        if let Some(desc) = high_risk_tld_indicator(url) {
            indicators.push(Indicator {
                category: IndicatorCategory::Domain,
                description: desc,
                weight: 0.5,
                source: IndicatorSource::Heuristic,
            });
        }
    }

    // Priority 4: Model n-gram indicators (supplementary, less interpretable)
    for (feature_idx, tree_count) in &ranked_features {
        if indicators.len() >= MAX_INDICATORS {
            break;
        }
        let feat_idx = *feature_idx;
        if (feat_idx as usize) < model.n_features {
            let bucket = feat_idx as usize;
            let ngrams = ngrams_for_bucket(url, bucket, model.n_features, model.ngram_range);
            if let Some(best) = ngrams.iter().max_by_key(|g| g.len()) {
                indicators.push(Indicator {
                    category: IndicatorCategory::NGram,
                    description: format!("Suspicious character pattern: '{}'", best),
                    weight: *tree_count as f32 / total_trees as f32,
                    source: IndicatorSource::Model {
                        tree_count: *tree_count,
                        total_trees,
                    },
                });
            }
        }
    }

    Prediction { score, indicators }
}

/// Check a manual feature for abnormality and return an indicator if abnormal.
///
/// Returns `Some((category, description))` if the feature value triggers a
/// risk indicator, `None` otherwise.
fn manual_feature_indicator(
    idx: usize,
    val: f32,
    url: &str,
) -> Option<(IndicatorCategory, String)> {
    match idx {
        // 0: has_http — excluded (protocol handled separately)
        // 1: has_https — not an indicator
        2 if val >= 4.0 => Some((
            IndicatorCategory::Structure,
            "URL length abnormal".to_string(),
        )),
        3 if val >= 3.0 => Some((
            IndicatorCategory::Domain,
            "Domain length abnormal".to_string(),
        )),
        4 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            "Uses IP address instead of domain name".to_string(),
        )),
        5 if val >= 3.0 => Some((
            IndicatorCategory::Domain,
            "Excessive subdomain levels".to_string(),
        )),
        7 if val >= 3.0 => Some((
            IndicatorCategory::Structure,
            "Excessive hyphens in URL".to_string(),
        )),
        9 if val >= 1.0 => Some((
            IndicatorCategory::Structure,
            "Contains @ symbol (URL obfuscation)".to_string(),
        )),
        10 if val >= 1.0 => Some((
            IndicatorCategory::Structure,
            "High percent-encoding usage".to_string(),
        )),
        14 if val > 0.3 => Some((
            IndicatorCategory::Structure,
            "High digit-to-character ratio".to_string(),
        )),
        15 if val >= 3.0 => Some((IndicatorCategory::Path, "Path length abnormal".to_string())),
        18 if val == 1.0 => {
            let keyword = find_sensitive_keyword(url);
            Some((
                IndicatorCategory::Path,
                if let Some(kw) = keyword {
                    format!("Contains sensitive keyword: '{}'", kw)
                } else {
                    "Contains sensitive keyword (login/verify/paypal)".to_string()
                },
            ))
        }
        _ => None,
    }
}

/// Find which sensitive keyword is present in the URL.
fn find_sensitive_keyword(url: &str) -> Option<&'static str> {
    let url_lower = url.to_lowercase();
    let words = [
        "login",
        "signin",
        "verify",
        "account",
        "password",
        "secure",
        "update",
        "bank",
        "paypal",
        "facebook",
        "google",
        "apple",
        "amazon",
        "ebay",
        "microsoft",
        "yahoo",
        "linkedin",
    ];
    words.iter().find(|&&w| url_lower.contains(w)).copied()
}

/// Check if the URL's TLD is in the high-risk list.
fn high_risk_tld_indicator(url: &str) -> Option<String> {
    let url_lower = url.to_lowercase();
    let cleaned = cleaned_domain(&url_lower);
    let tld = cleaned.split('.').next_back().unwrap_or("");
    let high_risk_tlds = [
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "xyz",
        "top",
        "click",
        "country",
        "stream",
        "download",
        "loan",
        "work",
        "men",
        "racing",
        "review",
        "party",
        "trade",
        "science",
        "accountant",
        "date",
        "cricket",
        "faith",
        "win",
    ];
    if high_risk_tlds.contains(&tld) {
        Some(format!("Uses high-risk TLD: .{}", tld))
    } else {
        None
    }
}

/// Extract the domain portion (before the first `/`) from a lowercased URL
/// with protocol stripped.
fn cleaned_domain(url_lower: &str) -> &str {
    let stripped = url_lower
        .strip_prefix("http://")
        .or_else(|| url_lower.strip_prefix("https://"))
        .unwrap_or(url_lower);
    match stripped.find('/') {
        Some(pos) => &stripped[..pos],
        None => stripped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_url_detailed_phishing() {
        let model = crate::load_default_model().expect("Failed to load model");
        let result = predict_url_detailed(
            "nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com",
            &model,
        );
        assert!(
            result.score >= 0.45,
            "Phishing URL should score >= 0.45, got {}",
            result.score
        );
        assert!(
            !result.indicators.is_empty(),
            "Phishing URL should have indicators"
        );
        assert!(
            result.indicators.len() <= MAX_INDICATORS,
            "Should have at most {} indicators, got {}",
            MAX_INDICATORS,
            result.indicators.len()
        );
        // Should have at least one Model-sourced indicator
        let has_model = result
            .indicators
            .iter()
            .any(|i| matches!(i.source, IndicatorSource::Model { .. }));
        assert!(has_model, "Should have at least one model indicator");
    }

    #[test]
    fn test_predict_url_detailed_normal() {
        let model = crate::load_default_model().expect("Failed to load model");
        let result = predict_url_detailed("http://example.com", &model);
        assert!(
            result.score < 0.45,
            "Normal URL should score < 0.45, got {}",
            result.score
        );
        assert!(
            result.indicators.len() <= MAX_INDICATORS,
            "Should have at most {} indicators",
            MAX_INDICATORS
        );
    }

    #[test]
    fn test_indicator_limit() {
        let model = crate::load_default_model().expect("Failed to load model");
        let urls = vec![
            "http://192.168.1.1/login/verify/account?password=123&token=abc",
            "http://very-long-phishing-url-with-many-subdomains.a.b.c.example.com/login",
        ];
        for url in &urls {
            let result = predict_url_detailed(url, &model);
            assert!(
                result.indicators.len() <= MAX_INDICATORS,
                "URL '{}' produced {} indicators (max {})",
                url,
                result.indicators.len(),
                MAX_INDICATORS
            );
        }
    }

    #[test]
    fn test_high_risk_tld_detection() {
        assert!(high_risk_tld_indicator("http://example.tk").is_some());
        assert!(high_risk_tld_indicator("http://phish.xyz").is_some());
        assert!(high_risk_tld_indicator("http://example.com").is_none());
    }

    #[test]
    fn test_cleaned_domain() {
        assert_eq!(cleaned_domain("http://example.com/path"), "example.com");
        assert_eq!(cleaned_domain("https://sub.example.com"), "sub.example.com");
        assert_eq!(cleaned_domain("example.com"), "example.com");
        assert_eq!(cleaned_domain("http://192.168.1.1/x"), "192.168.1.1");
    }

    #[test]
    fn test_find_sensitive_keyword() {
        assert_eq!(
            find_sensitive_keyword("http://example.com/login"),
            Some("login")
        );
        assert_eq!(
            find_sensitive_keyword("http://verify.account.com"),
            Some("verify")
        );
        assert_eq!(find_sensitive_keyword("http://example.com"), None);
    }

    #[test]
    fn test_ip_url_indicators() {
        let model = crate::load_default_model().expect("Failed to load model");
        let result = predict_url_detailed("http://192.168.1.1/login", &model);
        let has_ip_indicator = result
            .indicators
            .iter()
            .any(|i| i.description.contains("IP address"));
        assert!(
            has_ip_indicator,
            "IP-based URL should have IP address indicator, got: {:?}",
            result.indicators
        );
    }
}
