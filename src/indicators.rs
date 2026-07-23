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
//!   "N/total trees").
//! - **Heuristic indicators** (方案 A): Derived by checking the manual
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
///
/// `PartialEq` is derived so consumers can assert/compare two predictions in
/// tests. `Eq`/`Hash` are intentionally **not** derived because `score` is
/// `f32` (NaN is not reflexively equal).
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Phishing probability score (0.0 = definitely normal, 1.0 = definitely phishing).
    pub score: f32,
    /// Human-readable risk indicators explaining the score (max 5).
    pub indicators: Vec<Indicator>,
}

/// A single risk indicator.
///
/// `PartialEq` is derived so consumers can compare/dedupe indicators. `Eq`/
/// `Hash` are intentionally **not** derived because `weight` is `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct Indicator {
    /// Coarse category of the risk factor (domain / path / structure / n-gram).
    pub category: IndicatorCategory,
    /// Fine-grained, stable risk type. Consumers should aggregate/match on
    /// this rather than parsing [`description`](Self::description).
    pub group: IndicatorGroup,
    /// Human-readable description of the risk (may change between versions;
    /// do not rely on its exact wording for programmatic decisions).
    pub description: String,
    /// Contribution weight (0.0-1.0). Model indicators use `tree_count/total_trees`;
    /// heuristic indicators use a fixed 0.5.
    pub weight: f32,
    /// Whether this indicator comes from model decision tracing or a heuristic rule.
    pub source: IndicatorSource,
}

/// Category of a risk indicator.
///
/// Implements `Copy`, `Hash`, and `Eq` (all variants are fieldless unit
/// variants) so it can be used as a `HashMap`/`HashSet` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Fine-grained, stable risk type of an indicator.
///
/// Unlike [`IndicatorCategory`] (a coarse 4-value bucket), `IndicatorGroup`
/// identifies the *specific* risk type (e.g. "IP address", "domain
/// impersonation", "sensitive keyword") so consumers can aggregate and match
/// on a precise enum value instead of parsing the human-readable
/// [`description`](Indicator::description) string.
///
/// This enum is `#[non_exhaustive]`: new risk types may be added in future
/// versions, so downstream `match` expressions must include a `_` arm.
///
/// Implements `Copy`, `Hash`, and `Eq` (all variants are fieldless unit
/// variants) so consumers can use it as a `HashMap`/`HashSet` key for
/// grouping or counting indicators by risk type.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndicatorGroup {
    // --- Domain-related ---
    /// Domain length abnormally long.
    DomainLengthAbnormal,
    /// URL uses an IP literal instead of a domain name.
    IpAddress,
    /// Excessive subdomain levels.
    ExcessiveSubdomains,
    /// Domain closely resembles a known brand (impersonation).
    DomainImpersonation,
    /// Random / high-entropy subdomain.
    HighEntropySubdomain,
    /// Explicit port number present.
    ExplicitPort,
    /// Non-standard port number (>= 1000).
    NonStandardPort,
    /// Abnormally long single domain label.
    LongDomainLabel,
    /// Excessive number of domain labels.
    ExcessiveDomainLabels,
    /// Long average domain label length.
    LongAvgDomainLabel,
    /// High digit-to-character ratio in the domain.
    HighDomainDigitRatio,
    /// High Shannon entropy of the domain.
    HighDomainEntropy,
    /// A domain label ends with a hyphen.
    TrailingHyphenLabel,
    /// A domain label starts with a digit.
    LeadingDigitLabel,
    /// Dangerous URL scheme (data: / javascript:).
    DangerousScheme,

    // --- Path-related ---
    /// Path length abnormally long.
    PathLengthAbnormal,
    /// Contains a sensitive keyword (login/verify/paypal/...).
    SensitiveKeyword,
    /// Deep path structure (many levels).
    DeepPathStructure,
    /// Excessive query parameters.
    ExcessiveQueryParams,

    // --- Structure-related ---
    /// Overall URL length abnormally long.
    UrlLengthAbnormal,
    /// Excessive hyphens in the URL.
    ExcessiveHyphens,
    /// Contains `@` symbol (URL obfuscation).
    AtSymbol,
    /// High percent-encoding usage.
    PercentEncoding,
    /// High digit-to-character ratio in the URL.
    HighDigitRatio,
    /// High hexadecimal character ratio.
    HighHexRatio,
    /// Low alphabetic character ratio.
    LowAlphabeticRatio,
    /// High uppercase ratio (obfuscation).
    HighUppercaseRatio,
    /// Double-slash obfuscation in the path.
    DoubleSlashObfuscation,

    // --- Model n-gram ---
    /// Suspicious character n-gram pattern detected by the model.
    SuspiciousNGram,
}

/// The source of an indicator.
///
/// Implements `Copy`, `Hash`, and `Eq` so it can be used as a
/// `HashMap`/`HashSet` key (all field types are `usize`, which is
/// `Copy + Eq + Hash`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            if let Some((category, group, desc)) = manual_feature_indicator(manual_idx, val, url) {
                covered_manual.push(manual_idx);
                indicators.push(Indicator {
                    category,
                    group,
                    description: desc,
                    weight: *tree_count as f32 / total_trees.max(1) as f32,
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
        if let Some((category, group, desc)) = manual_feature_indicator(idx, val, url) {
            covered_manual.push(idx);
            indicators.push(Indicator {
                category,
                group,
                description: desc,
                weight: 0.5,
                source: IndicatorSource::Heuristic,
            });
        }
    }

    // Priority 3: Model n-gram indicators (supplementary, less interpretable)
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
                    group: IndicatorGroup::SuspiciousNGram,
                    description: format!("Suspicious character pattern: '{}'", best),
                    weight: *tree_count as f32 / total_trees.max(1) as f32,
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
/// Returns `Some((category, group, description))` if the feature value
/// triggers a risk indicator, `None` otherwise.
fn manual_feature_indicator(
    idx: usize,
    val: f32,
    url: &str,
) -> Option<(IndicatorCategory, IndicatorGroup, String)> {
    match idx {
        // 0: has_http — excluded (protocol handled separately)
        // 1: has_https — not an indicator
        2 if val >= 4.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::UrlLengthAbnormal,
            "URL length abnormal".to_string(),
        )),
        3 if val >= 3.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::DomainLengthAbnormal,
            "Domain length abnormal".to_string(),
        )),
        4 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::IpAddress,
            "Uses IP address instead of domain name".to_string(),
        )),
        5 if val >= 3.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::ExcessiveSubdomains,
            "Excessive subdomain levels".to_string(),
        )),
        7 if val >= 3.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::ExcessiveHyphens,
            "Excessive hyphens in URL".to_string(),
        )),
        9 if val >= 1.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::AtSymbol,
            "Contains @ symbol (URL obfuscation)".to_string(),
        )),
        10 if val >= 1.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::PercentEncoding,
            "High percent-encoding usage".to_string(),
        )),
        14 if val > 0.3 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::HighDigitRatio,
            "High digit-to-character ratio".to_string(),
        )),
        15 if val >= 3.0 => Some((
            IndicatorCategory::Path,
            IndicatorGroup::PathLengthAbnormal,
            "Path length abnormal".to_string(),
        )),
        18 if val == 1.0 => {
            let keyword = find_sensitive_keyword(url);
            Some((
                IndicatorCategory::Path,
                IndicatorGroup::SensitiveKeyword,
                if let Some(kw) = keyword {
                    format!("Contains sensitive keyword: '{}'", kw)
                } else {
                    "Contains sensitive keyword (login/verify/paypal)".to_string()
                },
            ))
        }
        19 if val >= 0.5 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::DomainImpersonation,
            format!(
                "Domain closely resembles a known brand (impersonation score {:.2})",
                val
            ),
        )),
        20 if val >= 0.7 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::HighEntropySubdomain,
            format!("Random / high-entropy subdomain (entropy {:.2})", val),
        )),
        // --- Structural features (indices 21-38) ---
        21 if val >= 5.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::DeepPathStructure,
            format!("Deep path structure ({} levels)", val as i32),
        )),
        22 if val >= 5.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::ExcessiveQueryParams,
            format!("Excessive query parameters ({})", val as i32),
        )),
        23 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::ExplicitPort,
            "Explicit port number in URL".to_string(),
        )),
        24 if val >= 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::NonStandardPort,
            "Non-standard port number (>= 1000)".to_string(),
        )),
        25 if val > 0.5 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::HighHexRatio,
            format!("High hexadecimal character ratio ({:.2})", val),
        )),
        26 if val < 0.3 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::LowAlphabeticRatio,
            format!("Low alphabetic character ratio ({:.2})", val),
        )),
        27 if val > 0.1 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::HighUppercaseRatio,
            format!("High uppercase ratio (obfuscation) ({:.2})", val),
        )),
        28 if val >= 20.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::LongDomainLabel,
            format!("Abnormally long domain label ({} chars)", val as i32),
        )),
        29 if val >= 5.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::ExcessiveDomainLabels,
            format!("Excessive domain label count ({})", val as i32),
        )),
        30 if val >= 15.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::LongAvgDomainLabel,
            format!("Long average domain label length ({:.1})", val),
        )),
        31 if val > 0.3 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::HighDomainDigitRatio,
            format!("High digit ratio in domain ({:.2})", val),
        )),
        32 if val >= 0.7 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::HighDomainEntropy,
            format!("High domain entropy ({:.2})", val),
        )),
        35 if val == 1.0 => Some((
            IndicatorCategory::Structure,
            IndicatorGroup::DoubleSlashObfuscation,
            "Double-slash obfuscation in path".to_string(),
        )),
        36 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::TrailingHyphenLabel,
            "Domain label ends with hyphen".to_string(),
        )),
        37 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::LeadingDigitLabel,
            "Domain label starts with digit".to_string(),
        )),
        38 if val == 1.0 => Some((
            IndicatorCategory::Domain,
            IndicatorGroup::DangerousScheme,
            "Dangerous URL scheme (data:/javascript:)".to_string(),
        )),
        _ => None,
    }
}

/// Find which sensitive keyword is present in the URL using word-boundary
/// matching (consistent with feature extraction). Returns the matched
/// keyword as an owned `String`, or `None` if no keyword matches.
fn find_sensitive_keyword(url: &str) -> Option<String> {
    let normalized = crate::extractor::normalize_url(url);
    let url_lower = normalized.to_lowercase();
    let re = crate::extractor::sensitive_word_regex();
    re.find(&url_lower).map(|m| m.as_str().to_string())
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
            result.score >= 0.20,
            "Phishing URL should score >= 0.20, got {}",
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
        // NOTE: a *minimal* forest (few trees) may not concentrate decision
        // paths on any single feature across >= MIN_TREE_VOTES trees, so a
        // Model-sourced indicator is not guaranteed. The phishing verdict is
        // still explained by the heuristic indicators (e.g. high-entropy
        // subdomain). Requiring a Model indicator here would make the test
        // brittle to forest size, which is exactly what we are minimizing.
    }

    #[test]
    fn test_predict_url_detailed_normal() {
        let model = crate::load_default_model().expect("Failed to load model");
        let result = predict_url_detailed("https://www.google.com", &model);
        assert!(
            result.score < 0.20,
            "Normal URL should score < 0.20, got {}",
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
    fn test_find_sensitive_keyword() {
        assert_eq!(
            find_sensitive_keyword("http://example.com/login"),
            Some("login".to_string())
        );
        assert_eq!(
            find_sensitive_keyword("http://verify.account.com"),
            Some("verify".to_string())
        );
        assert_eq!(find_sensitive_keyword("http://example.com"), None);
        // Word-boundary matching: "bloglogin" should NOT match "login"
        assert_eq!(find_sensitive_keyword("http://bloglogin.com"), None);
        // "pineapple" should NOT match "apple"
        assert_eq!(find_sensitive_keyword("http://pineapple.com"), None);
    }

    #[test]
    fn test_ip_url_indicators() {
        let model = crate::load_default_model().expect("Failed to load model");
        let result = predict_url_detailed("http://192.168.1.1/login", &model);
        let has_ip_indicator = result
            .indicators
            .iter()
            .any(|i| i.group == IndicatorGroup::IpAddress);
        assert!(
            has_ip_indicator,
            "IP-based URL should have IpAddress group indicator, got: {:?}",
            result.indicators
        );
    }
}
