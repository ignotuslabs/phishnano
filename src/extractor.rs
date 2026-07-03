//! Feature extraction module for URL analysis.
//!
//! This module transforms a raw URL string into a numerical feature vector
//! suitable for machine learning inference. The feature vector consists of
//! two parts:
//!
//! 1. **Character n-gram hash features** (indices 0 to `n_features - 1`):
//!    The URL is cleaned (protocol stripped, lowercased, digits normalized
//!    to "0"), then character n-grams of length 2-3 are extracted. Each
//!    n-gram is hashed using MurmurHash3 and mapped to a feature index via
//!    modulo operation. The count at each index represents how many n-grams
//!    hashed to that bucket.
//!
//! 2. **Manual engineered features** (indices `n_features` to
//!    `n_features + n_manual_features - 1`):
//!    19 hand-crafted features capturing URL structural properties such as
//!    length, special character counts, TLD category, and presence of
//!    sensitive keywords (e.g., "login", "verify", "paypal").
//!
//! # Feature Consistency
//!
//! The extraction logic in this module must remain identical to the Python
//! implementation in `training/scripts/train.py`. Any divergence will cause
//! the Rust inference results to differ from the Python training metrics.

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Matches the HTTP or HTTPS protocol prefix (e.g., "http://", "https://").
    static ref PROTOCOL_REGEX: Regex = Regex::new(r"^https?://").unwrap();

    /// Matches any single digit character (0-9) for digit normalization.
    static ref DIGIT_REGEX: Regex = Regex::new(r"\d").unwrap();

    /// Matches IPv4 address patterns in the domain portion of a URL.
    /// Used to detect URLs that use raw IP addresses instead of domain names,
    /// which is a common phishing indicator.
    static ref IP_REGEX: Regex = Regex::new(r"^\d+\.\d+\.\d+\.\d+").unwrap();
}

/// Extract the complete feature vector from a URL.
///
/// This function combines n-gram hash features and manual features into a
/// single flat vector. The layout is:
///
/// ```text
/// [n-gram hash features (n_features)] [manual features (n_manual_features)]
/// ```
///
/// # Arguments
///
/// - `url`: The raw URL string to analyze
/// - `n_features`: Number of n-gram hash buckets (typically 500)
/// - `n_manual_features`: Number of manual features (typically 19)
///
/// # Returns
///
/// A vector of `n_features + n_manual_features` floating-point values.
///
/// # N-gram Extraction Process
///
/// 1. The URL is cleaned via [`clean_url`] (strip protocol, lowercase, normalize digits)
/// 2. For each n in [2, 3], all character n-grams are extracted
/// 3. Each n-gram is hashed with MurmurHash3 (seed=0)
/// 4. The hash is mapped to a bucket via `hash % n_features`
/// 5. The count at each bucket is incremented
///
/// # Manual Features
///
/// See [`extract_manual_features`] for the list of 19 engineered features.
pub fn extract_features(url: &str, n_features: usize, n_manual_features: usize) -> Vec<f32> {
    let cleaned = clean_url(url);
    let chars: Vec<char> = cleaned.chars().collect();
    let mut features = vec![0.0; n_features + n_manual_features];

    // Extract character n-grams of length 2 and 3, hash each one, and
    // increment the count at the corresponding bucket index.
    for n in 2..=3 {
        for i in 0..=(chars.len() as i32 - n) {
            let gram: String = chars[i as usize..(i + n) as usize].iter().collect();
            let hash = murmurhash3::murmurhash3_x86_32(gram.as_bytes(), 0) as usize;
            let idx = hash % n_features;
            features[idx] += 1.0;
        }
    }

    // Append the 19 manual engineered features after the n-gram features.
    let manual_features = extract_manual_features(url);
    for (i, &val) in manual_features.iter().enumerate() {
        if i < n_manual_features {
            features[n_features + i] = val;
        }
    }

    features
}

/// Extract 19 manual engineered features from a URL.
///
/// These features capture structural and lexical properties of the URL that
/// are indicative of phishing attempts. Each feature is designed to be
/// computed quickly (O(n) in URL length) without network requests.
///
/// # Feature List
///
/// | Index | Feature               | Description |
/// |-------|-----------------------|-------------|
/// | 0     | `has_http`            | 1.0 if protocol is HTTP (insecure) |
/// | 1     | `has_https`           | 1.0 if protocol is HTTPS (secure) |
/// | 2     | `url_len_bucket`      | URL length bucket (1-5, after stripping protocol) |
/// | 3     | `domain_len_bucket`   | Domain length bucket (1-4) |
/// | 4     | `has_ip`              | 1.0 if domain is an IPv4 address |
/// | 5     | `num_subdomains`      | Subdomain count (capped at 5) |
/// | 6     | `num_dots`            | Total dot count in URL |
/// | 7     | `num_hyphens`         | Hyphen count in URL |
/// | 8     | `num_underscores`     | Underscore count in URL |
/// | 9     | `num_at`              | '@' count (used for URL obfuscation) |
/// | 10    | `num_percent`         | '%' count (URL encoding abuse) |
/// | 11    | `num_equals`          | '=' count (query parameter indicator) |
/// | 12    | `num_qmark`           | '?' count (query string start) |
/// | 13    | `num_and`             | '&' count (query parameter separator) |
/// | 14    | `digit_ratio`         | Ratio of digits to total characters |
/// | 15    | `path_len_bucket`     | Path length bucket (1-4) |
/// | 16    | `query_len_bucket`    | Query string length bucket (1-4) |
/// | 17    | `tld_code`            | TLD category code (0=other, 1-16=popular TLDs) |
/// | 18    | `has_sensitive_word`  | 1.0 if URL contains sensitive keywords |
///
/// # Sensitive Keywords
///
/// The following keywords trigger `has_sensitive_word = 1.0`:
/// `login`, `signin`, `verify`, `account`, `password`, `secure`,
/// `update`, `bank`, `paypal`, `facebook`, `google`, `apple`,
/// `amazon`, `ebay`, `microsoft`, `yahoo`, `linkedin`
///
/// # Arguments
///
/// - `url`: The raw URL string to analyze
///
/// # Returns
///
/// A vector of 19 floating-point feature values.
pub fn extract_manual_features(url: &str) -> Vec<f32> {
    let url_lower = url.to_lowercase();

    // Protocol detection: HTTP is insecure and commonly used in phishing.
    let has_http = if url_lower.starts_with("http://") {
        1.0
    } else {
        0.0
    };
    let has_https = if url_lower.starts_with("https://") {
        1.0
    } else {
        0.0
    };

    // Strip protocol prefix for structural analysis.
    let url_clean = PROTOCOL_REGEX.replace_all(&url_lower, "");
    let url_clean_str = url_clean.as_ref();

    // URL length bucketing: phishing URLs tend to be longer due to
    // embedded paths, subdomains, and tracking parameters.
    let url_len = url_clean_str.len();
    let url_len_bucket = if url_len < 30 {
        1.0
    } else if url_len < 60 {
        2.0
    } else if url_len < 100 {
        3.0
    } else if url_len < 150 {
        4.0
    } else {
        5.0
    };

    // Split domain and path at the first '/' character.
    let (domain, path) = match url_clean_str.find('/') {
        Some(pos) => (&url_clean_str[..pos], &url_clean_str[pos..]),
        None => (url_clean_str, ""),
    };

    // Domain length bucketing: unusually long domains may indicate
    // phishing (e.g., "account-verify-security-update.example.com").
    let domain_len = domain.len();
    let domain_len_bucket = if domain_len < 8 {
        1.0
    } else if domain_len < 15 {
        2.0
    } else if domain_len < 25 {
        3.0
    } else {
        4.0
    };

    // IP address detection: legitimate websites use domain names, not
    // raw IP addresses. Phishing URLs sometimes use IPs to evade
    // domain-based blocklists.
    let has_ip = if IP_REGEX.is_match(domain) { 1.0 } else { 0.0 };

    // Subdomain count: excessive subdomains (e.g., "a.b.c.d.e.com")
    // can indicate phishing attempts to mimic legitimate domains.
    let num_subdomains = domain.chars().filter(|&c| c == '.').count();
    let num_subdomains = std::cmp::min(num_subdomains, 5) as f32;

    // Special character counts: phishing URLs often contain unusual
    // characters for obfuscation, tracking, or parameter injection.
    let num_dots = url_clean_str.chars().filter(|&c| c == '.').count() as f32;
    let num_hyphens = url_clean_str.chars().filter(|&c| c == '-').count() as f32;
    let num_underscores = url_clean_str.chars().filter(|&c| c == '_').count() as f32;
    let num_at = url_clean_str.chars().filter(|&c| c == '@').count() as f32;
    let num_percent = url_clean_str.chars().filter(|&c| c == '%').count() as f32;
    let num_equals = url_clean_str.chars().filter(|&c| c == '=').count() as f32;
    let num_qmark = url_clean_str.chars().filter(|&c| c == '?').count() as f32;
    let num_and = url_clean_str.chars().filter(|&c| c == '&').count() as f32;

    // Digit ratio: phishing URLs often contain random digits or
    // numeric identifiers (e.g., "account123.verify456.com").
    let num_digits = url_clean_str
        .chars()
        .filter(|&c| c.is_ascii_digit())
        .count() as f32;
    let digit_ratio = num_digits / url_clean_str.len().max(1) as f32;

    // Path length bucketing: long paths with many segments can indicate
    // phishing (e.g., "/login/verify/account/update/confirm").
    let path_len = path.len();
    let path_len_bucket = if path_len == 0 {
        1.0
    } else if path_len < 20 {
        2.0
    } else if path_len < 50 {
        3.0
    } else {
        4.0
    };

    // Query string length bucketing: long query strings may contain
    // injected parameters or tracking data.
    let query_len = match path.find('?') {
        Some(pos) => path.len() - pos,
        None => 0,
    };
    let query_len_bucket = if query_len == 0 {
        1.0
    } else if query_len < 20 {
        2.0
    } else if query_len < 50 {
        3.0
    } else {
        4.0
    };

    // TLD categorization: map the top-level domain to a numeric code.
    // Popular TLDs get codes 1-16; unknown TLDs get code 0.
    // Phishing URLs often use unusual TLDs (.xyz, .top, .tk, etc.).
    let tld = domain.split('.').next_back().unwrap_or("");
    let popular_tlds = [
        "com", "org", "net", "edu", "gov", "io", "co", "me", "uk", "us", "cn", "jp", "de", "fr",
        "au", "ca",
    ];
    let tld_code = popular_tlds
        .iter()
        .position(|&s| s == tld)
        .map_or(0, |i| i + 1) as f32;

    // Sensitive keyword detection: phishing URLs frequently contain
    // words related to authentication, verification, or well-known brands.
    let sensitive_words = [
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
    let has_sensitive_word = if sensitive_words.iter().any(|&w| url_clean_str.contains(w)) {
        1.0
    } else {
        0.0
    };

    vec![
        has_http,
        has_https,
        url_len_bucket,
        domain_len_bucket,
        has_ip,
        num_subdomains,
        num_dots,
        num_hyphens,
        num_underscores,
        num_at,
        num_percent,
        num_equals,
        num_qmark,
        num_and,
        digit_ratio,
        path_len_bucket,
        query_len_bucket,
        tld_code,
        has_sensitive_word,
    ]
}

/// Clean a URL for n-gram feature extraction.
///
/// The cleaning process consists of three steps:
///
/// 1. **Strip protocol**: Remove `http://` or `https://` prefix
/// 2. **Lowercase**: Convert all characters to lowercase
/// 3. **Normalize digits**: Replace all digits (0-9) with "0"
///
/// Digit normalization reduces feature sparsity by treating all numeric
/// values as equivalent. For example, "example123.com" and "example456.com"
/// produce the same n-gram features after cleaning.
///
/// # Arguments
///
/// - `url`: The raw URL string to clean
///
/// # Returns
///
/// The cleaned URL string.
///
/// # Examples
///
/// ```
/// use phishnano::extractor::clean_url;
///
/// assert_eq!(
///     clean_url("https://Example123.com/path?query=123"),
///     "example000.com/path?query=000"
/// );
/// assert_eq!(clean_url("example.com"), "example.com");
/// ```
pub fn clean_url(url: &str) -> String {
    let s = PROTOCOL_REGEX.replace_all(url, "");
    let s = s.to_lowercase();
    let s = DIGIT_REGEX.replace_all(&s, "0");
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify URL cleaning: protocol stripping, lowercasing, and digit
    /// normalization are applied correctly.
    #[test]
    fn test_clean_url() {
        assert_eq!(
            clean_url("https://Example123.com/path?query=123"),
            "example000.com/path?query=000"
        );
        assert_eq!(
            clean_url("http://192.168.1.1/index.html"),
            "000.000.0.0/index.html"
        );
        assert_eq!(clean_url("example.com"), "example.com");
    }

    /// Verify that the feature vector has the correct length and contains
    /// non-zero values (indicating n-gram hashing is working).
    #[test]
    fn test_extract_features_basic() {
        let features = extract_features("example.com", 100, 19);
        assert_eq!(features.len(), 119);
        let sum: f32 = features.iter().sum();
        assert!(sum > 0.0);
    }

    /// Verify that manual features are correctly extracted, including
    /// protocol detection (HTTPS) and sensitive keyword detection ("login").
    #[test]
    fn test_extract_manual_features() {
        let features = extract_manual_features("https://example.com/login");
        assert_eq!(features.len(), 19);
        assert_eq!(features[0], 0.0); // has_http = false
        assert_eq!(features[1], 1.0); // has_https = true
        assert_eq!(features[18], 1.0); // has_sensitive_word = true ("login")
    }
}
