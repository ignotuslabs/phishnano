//! Feature extraction module for URL analysis.
//!
//! This module transforms a raw URL string into a numerical feature vector
//! suitable for machine learning inference. The feature vector consists of
//! two parts:
//!
//! 1. **Character n-gram hash features** (indices 0 to `n_features - 1`):
//!    The URL is cleaned (protocol stripped, lowercased, digits normalized
//!    to "0"), then character n-grams are extracted. Each n-gram is hashed
//!    using MurmurHash3 and mapped to a feature index via modulo operation.
//!    The count at each index represents how many n-grams hashed to that
//!    bucket.
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

use regex::Regex;
use std::sync::OnceLock;

/// Returns a static reference to the protocol-stripping regex.
///
/// Matches `http://` or `https://` at the start of the URL string.
fn protocol_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^https?://").unwrap())
}

/// Returns a static reference to the digit-matching regex.
///
/// Matches any single digit character (0-9) for digit normalization.
fn digit_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d").unwrap())
}

/// Returns a static reference to the IPv4-matching regex.
///
/// Matches raw IPv4 address patterns at the start of the domain portion.
fn ip_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d+\.\d+\.\d+\.\d+").unwrap())
}

/// MurmurHash3 x86 32-bit hash function.
///
/// Self-contained implementation of the MurmurHash3 algorithm (variant:
/// x86_32), compatible with the reference implementation by Austin Appleby.
/// Produces identical results to Python's `sklearn.utils.murmurhash3_32`
/// for the same input and seed.
fn murmurhash3_x86_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    let mut h1 = seed;
    let nblocks = data.len() / 4;

    for i in 0..nblocks {
        let k1 = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        let mut k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    if tail.len() >= 3 {
        k1 ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= data.len() as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;

    h1
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
/// - `ngram_range`: Character n-gram range `[min, max]` (typically `[2, 3]`)
///
/// # Returns
///
/// A vector of `n_features + n_manual_features` floating-point values.
///
/// # Panics
///
/// Panics if `n_features` is 0, as the modulo operation for hash bucket
/// assignment would cause a division by zero.
///
/// # N-gram Extraction Process
///
/// 1. The URL is cleaned via [`clean_url`] (strip protocol, lowercase, normalize digits)
/// 2. For each n in `[ngram_range[0], ngram_range[1]]`, all character n-grams are extracted
/// 3. Each n-gram is hashed with MurmurHash3 (seed=0)
/// 4. The hash is mapped to a bucket via `hash % n_features`
/// 5. The count at each bucket is incremented
///
/// # Manual Features
///
/// See [`extract_manual_features`] for the list of 19 engineered features.
///
/// # Examples
///
/// ```
/// use phishnano::extract_features;
///
/// let features = extract_features("https://example.com/login", 500, 19, [2, 3]);
/// assert_eq!(features.len(), 519);
/// ```
pub fn extract_features(
    url: &str,
    n_features: usize,
    n_manual_features: usize,
    ngram_range: [usize; 2],
) -> Vec<f32> {
    let cleaned = clean_url(url);
    let chars: Vec<char> = cleaned.chars().collect();
    let mut features = vec![0.0; n_features + n_manual_features];

    for n in ngram_range[0]..=ngram_range[1] {
        if n == 0 || n > chars.len() {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            let hash = murmurhash3_x86_32(gram.as_bytes(), 0) as usize;
            let idx = hash % n_features;
            features[idx] += 1.0;
        }
    }

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
/// # Known Limitations
///
/// - Sensitive words use substring matching (`contains`), which can produce
///   false positives (e.g., "pineapple" matches "apple"). This is consistent
///   with the Python training pipeline and the impact is diluted by the
///   519-dimensional feature vector.
/// - No percent-encoding or Punycode decoding is performed. This is a
///   known limitation shared with the Python training pipeline.
///
/// # Arguments
///
/// - `url`: The raw URL string to analyze
///
/// # Returns
///
/// A vector of 19 floating-point feature values.
///
/// # Examples
///
/// ```
/// use phishnano::extractor::extract_manual_features;
///
/// let features = extract_manual_features("https://example.com/login");
/// assert_eq!(features.len(), 19);
/// assert_eq!(features[1], 1.0);  // has_https = true
/// assert_eq!(features[18], 1.0); // has_sensitive_word = true ("login")
/// ```
pub fn extract_manual_features(url: &str) -> Vec<f32> {
    let url_lower = url.to_lowercase();

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

    let url_clean = protocol_regex().replace(&url_lower, "");
    let url_clean_str = url_clean.as_ref();

    let url_char_count = url_clean_str.chars().count();
    let url_len_bucket = if url_char_count < 30 {
        1.0
    } else if url_char_count < 60 {
        2.0
    } else if url_char_count < 100 {
        3.0
    } else if url_char_count < 150 {
        4.0
    } else {
        5.0
    };

    let (domain, path) = match url_clean_str.find('/') {
        Some(pos) => (&url_clean_str[..pos], &url_clean_str[pos..]),
        None => (url_clean_str, ""),
    };

    let domain_char_count = domain.chars().count();
    let domain_len_bucket = if domain_char_count < 8 {
        1.0
    } else if domain_char_count < 15 {
        2.0
    } else if domain_char_count < 25 {
        3.0
    } else {
        4.0
    };

    let has_ip = if ip_regex().is_match(domain) {
        1.0
    } else {
        0.0
    };

    let num_subdomains = domain.chars().filter(|&c| c == '.').count();
    let num_subdomains = std::cmp::min(num_subdomains, 5) as f32;

    let num_dots = url_clean_str.chars().filter(|&c| c == '.').count() as f32;
    let num_hyphens = url_clean_str.chars().filter(|&c| c == '-').count() as f32;
    let num_underscores = url_clean_str.chars().filter(|&c| c == '_').count() as f32;
    let num_at = url_clean_str.chars().filter(|&c| c == '@').count() as f32;
    let num_percent = url_clean_str.chars().filter(|&c| c == '%').count() as f32;
    let num_equals = url_clean_str.chars().filter(|&c| c == '=').count() as f32;
    let num_qmark = url_clean_str.chars().filter(|&c| c == '?').count() as f32;
    let num_and = url_clean_str.chars().filter(|&c| c == '&').count() as f32;

    let num_digits = url_clean_str
        .chars()
        .filter(|&c| c.is_ascii_digit())
        .count() as f32;
    let digit_ratio = num_digits / url_char_count.max(1) as f32;

    let path_char_count = path.chars().count();
    let path_len_bucket = if path_char_count == 0 {
        1.0
    } else if path_char_count < 20 {
        2.0
    } else if path_char_count < 50 {
        3.0
    } else {
        4.0
    };

    let query_len = match path.find('?') {
        Some(pos) => path.chars().count() - path[..pos].chars().count(),
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

    let tld = domain.split('.').next_back().unwrap_or("");
    let popular_tlds = [
        "com", "org", "net", "edu", "gov", "io", "co", "me", "uk", "us", "cn", "jp", "de", "fr",
        "au", "ca",
    ];
    let tld_code = popular_tlds
        .iter()
        .position(|&s| s == tld)
        .map_or(0, |i| i + 1) as f32;

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
    let s = protocol_regex().replace(url, "");
    let s = s.to_lowercase();
    let s = digit_regex().replace_all(&s, "0");
    s.to_string()
}

/// Reverse-map an n-gram hash bucket to the actual n-gram(s) from a URL.
///
/// Given a bucket index (feature index in the n-gram hash space), this
/// function re-derives which character n-gram(s) from the URL hashed to
/// that bucket. This is used by
/// [`predict_url_detailed`](crate::indicators::predict_url_detailed) to
/// translate model decision paths into human-readable risk indicators.
///
/// # Arguments
///
/// - `url`: The raw URL string (same one passed to [`extract_features`])
/// - `bucket`: The n-gram hash bucket index to look up
/// - `n_features`: Number of n-gram hash buckets (must match `extract_features`)
/// - `ngram_range`: Character n-gram range `[min, max]`
///
/// # Returns
///
/// A vector of n-gram strings from the URL that hashed to the given bucket.
/// The vector may be empty (no n-gram hit this bucket), contain one entry,
/// or contain multiple entries (hash collisions).
pub fn ngrams_for_bucket(
    url: &str,
    bucket: usize,
    n_features: usize,
    ngram_range: [usize; 2],
) -> Vec<String> {
    let cleaned = clean_url(url);
    let chars: Vec<char> = cleaned.chars().collect();
    let mut result = Vec::new();

    for n in ngram_range[0]..=ngram_range[1] {
        if n == 0 || n > chars.len() {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            let hash = murmurhash3_x86_32(gram.as_bytes(), 0) as usize;
            if hash % n_features == bucket {
                result.push(gram);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_murmurhash3_deterministic() {
        let h1 = murmurhash3_x86_32(b"hello", 0);
        let h2 = murmurhash3_x86_32(b"hello", 0);
        assert_eq!(h1, h2, "Same input must produce same hash");

        let h3 = murmurhash3_x86_32(b"world", 0);
        assert_ne!(h1, h3, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_murmurhash3_empty() {
        let h = murmurhash3_x86_32(b"", 0);
        assert_eq!(h, 0, "Empty input with seed 0 should hash to 0");
    }

    #[test]
    fn test_murmurhash3_seed_sensitivity() {
        let h1 = murmurhash3_x86_32(b"hello", 0);
        let h2 = murmurhash3_x86_32(b"hello", 1);
        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_extract_features_basic() {
        let features = extract_features("example.com", 100, 19, [2, 3]);
        assert_eq!(features.len(), 119);
        let sum: f32 = features.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_extract_manual_features() {
        let features = extract_manual_features("https://example.com/login");
        assert_eq!(features.len(), 19);
        assert_eq!(features[0], 0.0);
        assert_eq!(features[1], 1.0);
        assert_eq!(features[18], 1.0);
    }

    #[test]
    fn test_ngrams_for_bucket() {
        // "example.com" cleaned = "example.com"
        // 2-grams: ex, xa, am, mp, pl, le, e., .c, co, om
        // Find which bucket "lo" would NOT be in (it's not in "example.com")
        let grams = ngrams_for_bucket("example.com", 0, 500, [2, 3]);
        // Bucket 0 may or may not have hits, but the function should not panic
        // and should return valid strings
        for g in &grams {
            assert!(!g.is_empty());
        }

        // Verify that ngrams_for_bucket returns the same n-grams that
        // extract_features counted in that bucket
        let features = extract_features("example.com", 500, 0, [2, 3]);
        for (bucket, &count) in features.iter().enumerate() {
            let ngrams = ngrams_for_bucket("example.com", bucket, 500, [2, 3]);
            assert_eq!(
                count as usize,
                ngrams.len(),
                "Bucket {} count mismatch",
                bucket
            );
        }
    }

    #[test]
    fn test_ngrams_for_bucket_empty_url() {
        let grams = ngrams_for_bucket("", 0, 500, [2, 3]);
        assert!(grams.is_empty(), "Empty URL should produce no n-grams");
    }
}
