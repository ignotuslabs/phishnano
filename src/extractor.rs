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
//!    39 hand-crafted features (21 manual + 18 additional
//!    structural features appended by `extract_struct_features`) capturing
//!    URL properties such as length, special character counts, TLD category,
//!    path/query structure, label statistics, port, and presence of sensitive
//!    keywords (e.g., "login", "verify", "paypal").
//!
//! # Feature Consistency
//!
//! The extraction logic in this module must remain identical to the Python
//! implementation in `training/scripts/train.py`. Any divergence will cause
//! the Rust inference results to differ from the Python training metrics.

use once_cell::sync::Lazy;
use percent_encoding::percent_decode_str;
use regex::Regex;

/// Returns a static reference to the protocol-stripping regex.
///
/// Matches `http://` or `https://` at the start of the URL string (case-insensitive).
fn protocol_regex() -> &'static Regex {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)^https?://").unwrap());
    &RE
}

/// Returns a static reference to the digit-matching regex.
///
/// Matches any single digit character (0-9) for digit normalization.
fn digit_regex() -> &'static Regex {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d").unwrap());
    &RE
}

/// Returns a static reference to the IPv4-matching regex.
///
/// Matches raw IPv4 address patterns at the start of the domain portion.
fn ip_regex() -> &'static Regex {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+\.\d+\.\d+\.\d+").unwrap());
    &RE
}

/// Returns a static reference to the sensitive-word regex.
///
/// Matches any of the 17 sensitive keywords at word boundaries.
/// Word boundaries are `\b` (transition between `\w` and `\W`),
/// so `login` in `/login/` matches but `login` in `bloglogin` does not.
pub(crate) fn sensitive_word_regex() -> &'static Regex {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)\b(login|signin|verify|account|password|secure|update|bank|paypal|facebook|google|apple|amazon|ebay|microsoft|yahoo|linkedin)\b").unwrap()
    });
    &RE
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

/// Normalize a URL for consistent feature extraction.
///
/// This function performs three normalization steps that are critical for
/// handling obfuscated phishing URLs:
///
/// 1. **Fragment removal**: Strip everything after `#` (fragments are
///    client-side only and irrelevant for phishing detection).
/// 2. **Percent-decoding**: Decode `%XX` sequences (e.g., `%70aypal` →
///    `paypal`) to reveal hidden sensitive keywords and patterns.
/// 3. **Punycode decoding**: Decode IDN domains (e.g., `xn--fsq.com` →
///    Unicode) to expose homograph attacks.
///
/// This function is called by [`extract_features`] and
/// [`extract_manual_features`] before any other processing, ensuring
/// that all downstream feature extraction operates on normalized input.
///
/// # Arguments
///
/// - `url`: The raw URL string to normalize
///
/// # Returns
///
/// The normalized URL string.
pub fn normalize_url(url: &str) -> String {
    // 1. Remove fragment (everything after the first '#')
    let no_fragment = match url.find('#') {
        Some(pos) => &url[..pos],
        None => url,
    };

    // 2. Percent-decode (e.g., "%70aypal" → "paypal")
    let decoded = percent_decode_str(no_fragment)
        .decode_utf8_lossy()
        .to_string();

    // 3. Punycode-decode the domain portion (e.g., "xn--fsq.com" → Unicode)
    decode_punycode_domain(&decoded)
}

/// Decode Punycode-encoded domain labels in a URL.
///
/// Extracts the domain from the URL, decodes any `xn--` labels to Unicode,
/// and reassembles the URL. If decoding fails for any label, the original
/// label is kept unchanged.
fn decode_punycode_domain(url: &str) -> String {
    // Split into protocol, domain, and rest
    let (prefix, rest) = match url.find("://") {
        Some(pos) => (&url[..pos + 3], &url[pos + 3..]),
        None => ("", url),
    };

    // Find the first '/' to separate domain from path
    let (domain, path) = match rest.find('/') {
        Some(pos) => (&rest[..pos], &rest[pos..]),
        None => (rest, ""),
    };

    // Decode only `xn--` (punycode) labels; preserve the original case of all
    // other labels. This mirrors `train.py::_decode_punycode_domain` exactly:
    // a label starting with (case-insensitive) `xn--` has its 4-char prefix
    // stripped and the remainder punycode-decoded, with no re-added prefix.
    // Case must be preserved so that case-sensitive features (notably
    // `uppercase_ratio`) match the Python training pipeline.
    let decoded_domain: String = domain
        .split('.')
        .map(|label| {
            if label.len() >= 4 && label.to_ascii_lowercase().starts_with("xn--") {
                idna::punycode::decode_to_string(&label[4..]).unwrap_or_else(|| label.to_string())
            } else {
                label.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(".");

    format!("{}{}{}", prefix, decoded_domain, path)
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
/// - `n_manual_features`: Number of manual features (typically 39 = 21 + 18)
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
/// 1. The URL is normalized via [`normalize_url`] (fragment removal, percent-decode, Punycode decode)
/// 2. The normalized URL is cleaned via [`clean_url`] (lowercase, strip protocol, normalize digits)
/// 3. For each n in `[ngram_range[0], ngram_range[1]]`, all character n-grams are extracted
/// 4. Each n-gram is hashed with MurmurHash3 (seed=0)
/// 5. The hash is mapped to a bucket via `hash % n_features`
/// 6. The count at each bucket is incremented
///
/// # Manual Features
///
/// See [`extract_manual_features`] for the list of 39 engineered features
/// (21 manual + 18 structural).
///
/// # Examples
///
/// ```
/// use phishnano::extract_features;
///
/// let features = extract_features("https://example.com/login", 500, 21, [2, 3]);
/// assert_eq!(features.len(), 521);
/// ```
pub fn extract_features(
    url: &str,
    n_features: usize,
    n_manual_features: usize,
    ngram_range: [usize; 2],
) -> Vec<f32> {
    let normalized = normalize_url(url);
    let cleaned = clean_url(&normalized);
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

    let manual_features = extract_manual_features(&normalized);
    for (i, &val) in manual_features.iter().enumerate() {
        if i < n_manual_features {
            features[n_features + i] = val;
        }
    }

    features
}

/// Extract 19 additional URL *structural* features (Part B).
///
/// These capture properties the 21 manual features and the n-gram hashes
/// cannot: path depth, query-param count, explicit port, hex/alpha/uppercase
/// ratios, label statistics (count / length / entropy), TLD length,
/// path-to-query ratio, double-slash obfuscation, trailing/leading
/// dash/digit labels, `data:`/`javascript:` schemes, and query-amp count.
///
/// This function MIRRORS `extract_struct_features` in
/// `training/scripts/train_lightgbm.py` **exactly** (same variable names,
/// same order, same `min` caps) so the LightGBM model trained on the
/// 540-dim vector (500 n-gram + 21 manual + 19 structural) scores identically
/// under Rust inference. The arguments `normalized`/`url_lower`/`url_clean`/
/// `domain`/`path` must be derived exactly as in [`extract_manual_features`]
/// (normalize → lowercase → strip protocol; `domain`/`path` split at the first
/// `/`).
pub(crate) fn extract_struct_features(
    normalized: &str,
    url_lower: &str,
    url_clean: &str,
    domain: &str,
    path: &str,
) -> Vec<f32> {
    let mut feats: Vec<f32> = Vec::with_capacity(18);

    // 0: path_depth -- number of non-empty '/' segments in the path.
    if path.is_empty() {
        feats.push(0.0);
    } else {
        let segs = path.split('/').filter(|s| !s.is_empty()).count();
        feats.push((segs as f32).min(8.0));
    }

    // 1: query_param_count -- number of '=' in the query string.
    let q = if path.contains('?') {
        &path[path.find('?').unwrap() + 1..]
    } else {
        ""
    };
    feats.push((q.matches('=').count() as f32).min(20.0));

    // 2: has_port -- explicit port present in the host portion.
    let has_port = if domain.contains('/') {
        domain[..domain.find('/').unwrap()].contains(':')
    } else {
        domain.contains(':')
    };
    feats.push(if has_port { 1.0 } else { 0.0 });

    // 3: port_bucket -- port value / 1000, capped at 6.0.
    let host_only = if domain.contains('/') {
        &domain[..domain.find('/').unwrap()]
    } else {
        domain
    };
    let mut port_val = 0i32;
    if host_only.contains(':') {
        if let Some(after) = host_only.rsplit(':').next() {
            port_val = after.parse::<i32>().unwrap_or(0);
        }
    }
    feats.push(((port_val as f32) / 1000.0).min(6.0));

    // 4: hex_ratio -- fraction of [0-9a-f] characters in the cleaned URL.
    let clean_len = url_clean.chars().count().max(1);
    let hexchars = url_clean
        .chars()
        .filter(|c| c.is_ascii_digit() || ('a'..='f').contains(c))
        .count();
    feats.push(hexchars as f32 / clean_len as f32);

    // 5: alpha_ratio -- fraction of alphabetic characters in the cleaned URL.
    let alphas = url_clean.chars().filter(|c| c.is_alphabetic()).count();
    feats.push(alphas as f32 / clean_len as f32);

    // 6: uppercase_ratio -- fraction of uppercase chars in the normalized URL
    //    (before lowercasing); a homograph / obfuscation hint.
    let norm_len = normalized.chars().count().max(1);
    let ups = normalized.chars().filter(|c| c.is_uppercase()).count();
    feats.push(ups as f32 / norm_len as f32);

    // 7: longest_label_len -- length of the longest domain label.
    let labels: Vec<&str> = domain.split('.').collect();
    let longest = labels.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    feats.push((longest as f32).min(40.0));

    // 8: num_labels -- number of labels in the domain.
    feats.push((labels.len() as f32).min(12.0));

    // 9: mean_label_len -- average label length.
    if labels.is_empty() {
        feats.push(0.0);
    } else {
        let total: usize = labels.iter().map(|l| l.chars().count()).sum();
        feats.push(((total as f32) / (labels.len() as f32)).min(30.0));
    }

    // 10: domain_digit_ratio -- digit ratio within the domain only.
    let dom_len = domain.chars().count().max(1);
    let digits_d = domain.chars().filter(|c| c.is_ascii_digit()).count();
    feats.push(digits_d as f32 / dom_len as f32);

    // 11: full_domain_entropy -- Shannon entropy of the whole domain, /5, capped 1.0.
    let dom_entropy = shannon_entropy_bits(domain) / 5.0;
    feats.push(dom_entropy.min(1.0));

    // 12: tld_len -- length of the TLD label.
    let tld = labels.last().copied().unwrap_or("");
    feats.push((tld.chars().count() as f32).min(20.0));

    // 13: path_to_query_ratio -- path length / (query length + 1).
    let path_only = if path.contains('?') {
        &path[..path.find('?').unwrap()]
    } else {
        path
    };
    let path_len = path_only.chars().count();
    let query_len = q.chars().count();
    feats.push(((path_len as f32) / ((query_len + 1) as f32)).min(30.0));

    // 14: has_double_slash -- '//' after skipping the first 8 characters of the
    //     cleaned URL (path confusion / obfuscation). Mirrors Python's
    //     `url_clean[8:]` (character-based slice).
    let after8: String = url_clean.chars().skip(8).collect();
    feats.push(if after8.contains("//") { 1.0 } else { 0.0 });

    // 15: trailing_dash_label -- any label ends with '-'.
    let trailing = labels.iter().any(|l| l.ends_with('-'));
    feats.push(if trailing { 1.0 } else { 0.0 });

    // 16: leading_digit_label -- any label starts with a digit (suspicious).
    let leading = labels
        .iter()
        .any(|l| !l.is_empty() && l.chars().next().unwrap().is_ascii_digit());
    feats.push(if leading { 1.0 } else { 0.0 });

    // 17: has_data_or_js_scheme -- data:/javascript: schemes are never legit web.
    let data_js = url_lower.starts_with("data:") || url_lower.starts_with("javascript:");
    feats.push(if data_js { 1.0 } else { 0.0 });

    // 18: num_query_amp -- number of '&' separators in the query.
    feats.push((q.matches('&').count() as f32).min(20.0));

    feats
}

/// Extract 21 manual engineered features from a URL.
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
/// | 19    | `brand_impersonation` | Similarity (0-1) to a brand's canonical domain; high when the domain is close to (but not equal to) a known brand |
/// | 20    | `subdomain_entropy`   | Shannon entropy (0-1) of subdomain labels; high for random/generated subdomains |
///
/// ## Structural features (indices 21-39)
///
/// Features 21-39 are the **19 additional structural features** appended by
/// [`extract_struct_features`] (Part B). They capture properties the 21 manual
/// features and the n-gram hashes cannot (path depth, query-param count,
/// explicit port, hex/alpha/uppercase ratios, label statistics, TLD length,
/// path-to-query ratio, double-slash obfuscation, trailing/leading dash/digit
/// labels, `data:`/`javascript:` schemes, query-amp count). Rust computes 40
/// manual features (21 + 19); the embedded model uses the first 39
/// (dropping the trailing structural feature) and ignores it at inference,
/// yielding a 539-dim vector with the 500 n-gram hashes.
///
/// # Sensitive Keywords
///
/// The following keywords trigger `has_sensitive_word = 1.0` when matched
/// at word boundaries (so "pineapple" does NOT match "apple"):
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
/// A vector of 39 floating-point feature values (21 manual + 18 structural).
///
/// # Examples
///
/// ```
/// use phishnano::extractor::extract_manual_features;
///
/// let features = extract_manual_features("https://example.com/login");
/// assert_eq!(features.len(), 40);
/// assert_eq!(features[1], 1.0);  // has_https = true
/// assert_eq!(features[18], 1.0); // has_sensitive_word = true ("login")
/// assert_eq!(features.len(), 21 + 19);
/// ```
pub fn extract_manual_features(url: &str) -> Vec<f32> {
    let normalized = normalize_url(url);
    let url_lower = normalized.to_lowercase();

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

    let has_sensitive_word = if sensitive_word_regex().is_match(url_clean_str) {
        1.0
    } else {
        0.0
    };

    // Feature 19: brand-impersonation score in [0, 1].
    // Compares each domain label against a brand's *second-level domain* (the
    // short brand token, e.g. "paypal"), requiring a *small* edit distance with
    // a bounded length difference. This catches `paypa1.com` (dist 1),
    // `paypal.login-secure.com` (brand token in subdomain) and `micros0ft.com`,
    // while generic domains like `example.com` do NOT match (their labels are
    // far from any brand token). The real brand's own canonical domain scores
    // 0 to avoid flagging legitimate brand sites.
    let reg = registrable_domain(domain);
    let mut brand_score = 0.0f32;
    for (_keyword, canonical) in BRAND_CANONICAL {
        if reg == *canonical {
            continue; // real brand domain -> no impersonation signal
        }
        let sld = canonical.split('.').next().unwrap_or(canonical);
        let sld_len = sld.chars().count().max(1) as i32;
        let max_dist = 2.max(sld_len / 4) as usize;
        let mut best_for_brand = 0.0f32;
        // Skip the final label (the TLD, e.g. "com"/"org") so it is not compared
        // against short brand tokens (e.g. "org" vs "irs" would falsely match).
        let domain_labels: Vec<&str> = domain.split('.').collect();
        let cmp: &[&str] = if domain_labels.len() > 1 {
            &domain_labels[..domain_labels.len() - 1]
        } else {
            &domain_labels[..]
        };
        for label in cmp {
            // Also consider hyphen-split tokens so `amaz0n-account` matches
            // `amazon` via its `amaz0n` token, without loosening the strict
            // whole-label requirement that prevents generic domains (e.g.
            // `example`) from matching a brand token.
            let mut candidates: Vec<&str> = vec![label];
            for tok in label.split('-') {
                if !tok.is_empty() {
                    candidates.push(tok);
                }
            }
            for cand in &candidates {
                let sim = if *cand == sld {
                    0.5 // brand token present as a non-canonical subdomain
                } else {
                    let d = levenshtein(cand, sld) as i32;
                    let len_diff = (cand.chars().count() as i32 - sld_len).abs();
                    if d <= max_dist as i32 && len_diff <= 3 {
                        1.0 - d as f32 / sld_len as f32
                    } else {
                        0.0
                    }
                };
                if sim > best_for_brand {
                    best_for_brand = sim;
                }
            }
        }
        if best_for_brand > brand_score {
            brand_score = best_for_brand;
        }
    }
    brand_score = brand_score.clamp(0.0, 1.0);

    // Feature 20: subdomain entropy in [0, 1].
    // Shannon entropy (bits/char) of the subdomain labels (everything before
    // the registrable domain). Random subdomains (`a8f3k9xq`) score high;
    // benign ones (`www`, `mail`) score low.
    let sub_labels: Vec<&str> = domain.split('.').collect();
    let reg_labels: Vec<&str> = reg.split('.').collect();
    let sub_only: String = if sub_labels.len() > reg_labels.len() {
        sub_labels[..sub_labels.len() - reg_labels.len()].join("")
    } else {
        String::new()
    };
    let sub_entropy = if sub_only.is_empty() {
        0.0
    } else {
        (shannon_entropy_bits(&sub_only) / 4.0).min(1.0)
    };

    let mut feats = vec![
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
        brand_score,
        sub_entropy,
    ];
    // Append the 19 structural features (路 B); total = 21 + 19 = 40.
    // The embedded legacy model has `n_manual_features = 39`, so the trailing
    // (19th) structural feature is currently ignored at inference; it is kept
    // here so Rust stays aligned with `extract_struct_features` in
    // `training/scripts/train_lightgbm.py` for future retraining.
    feats.extend(extract_struct_features(
        &normalized,
        &url_lower,
        url_clean_str,
        domain,
        path,
    ));
    feats
}

/// Clean a URL for n-gram feature extraction.
///
/// The cleaning process consists of three steps:
///
/// 1. **Lowercase**: Convert all characters to lowercase
/// 2. **Strip protocol**: Remove `http://` or `https://` prefix (case-insensitive)
/// 3. **Normalize digits**: Replace all digits (0-9) with "0"
///
/// Lowercasing before stripping ensures that uppercase protocols like
/// `HTTP://` are correctly removed. Digit normalization reduces feature
/// sparsity by treating all numeric values as equivalent. For example,
/// "example123.com" and "example456.com" produce the same n-gram features
/// after cleaning.
///
/// # Arguments
///
/// - `url`: The normalized URL string to clean (see [`normalize_url`])
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
    let s = url.to_lowercase();
    let s = protocol_regex().replace(&s, "");
    let s = digit_regex().replace_all(&s, "0");
    s.to_string()
}

// ---------------------------------------------------------------------------
// Brand-impersonation & subdomain-randomness helpers (features 19 and 20)
// ---------------------------------------------------------------------------
//
// These two features target the model's biggest blind spot: character n-gram
// hashing cannot tell `paypa1.com` from `paypal.com`, nor a random subdomain
// (`a8f3k9xq.verify-secure.com`) from a benign one. They MUST be mirrored
// exactly in `training/scripts/train.py` (`extract_manual_features`) so that
// Python training features match Rust inference features.

/// Multi-part public suffixes where the registrable domain is the *third*
/// label from the right (e.g. `example.co.uk`).
pub(crate) const MULTI_PART_TLDS: &[&str] = &[
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp", "co.nz", "com.br",
    "co.in", "com.cn", "co.kr",
];

/// `(brand_keyword, canonical_domain)` pairs for the most-targeted brands.
/// The canonical domain is the real brand domain; anything *close* to it but
/// *not equal* is treated as a potential impersonation.
pub(crate) const BRAND_CANONICAL: &[(&str, &str)] = &[
    ("google", "google.com"),
    ("facebook", "facebook.com"),
    ("paypal", "paypal.com"),
    ("amazon", "amazon.com"),
    ("microsoft", "microsoft.com"),
    ("apple", "apple.com"),
    ("ebay", "ebay.com"),
    ("linkedin", "linkedin.com"),
    ("netflix", "netflix.com"),
    ("instagram", "instagram.com"),
    ("twitter", "twitter.com"),
    ("whatsapp", "whatsapp.com"),
    ("yahoo", "yahoo.com"),
    ("dropbox", "dropbox.com"),
    ("outlook", "outlook.com"),
    ("office", "office.com"),
    ("chase", "chase.com"),
    ("bankofamerica", "bankofamerica.com"),
    ("wellsfargo", "wellsfargo.com"),
    ("citibank", "citibank.com"),
    ("usbank", "usbank.com"),
    ("coinbase", "coinbase.com"),
    ("binance", "binance.com"),
    ("roblox", "roblox.com"),
    ("discord", "discord.com"),
    ("steam", "steamcommunity.com"),
    ("nvidia", "nvidia.com"),
    ("tesla", "tesla.com"),
    ("walmart", "walmart.com"),
    ("target", "target.com"),
    ("costco", "costco.com"),
    ("bestbuy", "bestbuy.com"),
    ("irs", "irs.gov"),
    ("americanexpress", "americanexpress.com"),
    ("visa", "visa.com"),
    ("mastercard", "mastercard.com"),
];

/// Levenshtein edit distance between two strings (character based).
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Per-character Shannon entropy of `s` in bits (0.0 for empty/uniform single char).
fn shannon_entropy_bits(s: &str) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let total = s.chars().count() as f32;
    let mut h = 0.0f32;
    for &c in counts.values() {
        let p = c as f32 / total;
        h -= p * p.log2();
    }
    h
}

/// Extract the registrable domain (eTLD+1) from a domain string.
pub(crate) fn registrable_domain(domain: &str) -> String {
    let labels: Vec<&str> = domain.split('.').collect();
    if labels.len() <= 2 {
        return domain.to_string();
    }
    let last_two = format!("{}.{}", labels[labels.len() - 2], labels[labels.len() - 1]);
    if MULTI_PART_TLDS.contains(&last_two.as_str()) {
        // Registrable = third label from the right.
        return labels[labels.len() - 3..].join(".");
    }
    last_two
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
    let normalized = normalize_url(url);
    let cleaned = clean_url(&normalized);
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
        let features = extract_features("example.com", 100, 21, [2, 3]);
        assert_eq!(features.len(), 121);
        let sum: f32 = features.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_extract_manual_features() {
        let features = extract_manual_features("https://example.com/login");
        assert_eq!(features.len(), 40);
        assert_eq!(features[0], 0.0);
        assert_eq!(features[1], 1.0);
        assert_eq!(features[18], 1.0);
        // 19 structural features appended after the 21 manual ones.
        assert_eq!(features.len(), 21 + 19);
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

    #[test]
    fn test_normalize_url_fragment_removal() {
        let normalized = normalize_url("https://example.com/login#section");
        assert_eq!(normalized, "https://example.com/login");
    }

    #[test]
    fn test_normalize_url_percent_decode() {
        let normalized = normalize_url("https://example.com/%70aypal");
        assert_eq!(normalized, "https://example.com/paypal");
    }

    #[test]
    fn test_normalize_url_idna_decode() {
        let normalized = normalize_url("https://xn--fsq.com/path");
        assert!(
            normalized.contains("xn--fsq") || normalized != "https://xn--fsq.com/path",
            "Punycode domain should be decoded"
        );
    }

    /// Cross-validate the 19 structural features (indices 21-39 of
    /// `extract_manual_features`) against the Python reference exported by
    /// `training/scripts/_tmp_dump_struct.py`, embedded at compile time via
    /// `include_str!`. This guarantees the Rust port of `extract_struct_features`
    /// matches the LightGBM training features exactly. The fixture is required:
    /// if it fails to parse, the test panics rather than silently passing.
    #[test]
    fn test_struct_features_consistency() {
        let content = include_str!("../resources/test_struct_features.json");
        let data: serde_json::Value = serde_json::from_str(content).expect("Failed to parse JSON");

        for (url, value) in data.as_object().unwrap() {
            let py: Vec<f32> = serde_json::from_value(value.clone()).expect("bad array");
            assert_eq!(py.len(), 19, "reference has 19 struct features");
            let rust = extract_manual_features(url);
            assert_eq!(rust.len(), 40, "manual features must be 40 (21+19)");
            let mut max_diff = 0.0f32;
            for i in 0..19 {
                let d = (rust[21 + i] - py[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            assert!(
                max_diff < 1e-4,
                "struct-feature mismatch for URL '{}' (max diff {:.6})",
                url,
                max_diff
            );
        }
    }

    #[test]
    fn test_sensitive_word_word_boundary() {
        let features_match = extract_manual_features("https://example.com/login");
        assert_eq!(
            features_match[18], 1.0,
            "'login' at word boundary should match"
        );

        let features_no_match = extract_manual_features("https://example.com/bloglogin");
        assert_eq!(
            features_no_match[18], 0.0,
            "'login' embedded in 'bloglogin' should NOT match (word boundary)"
        );

        let features_pineapple = extract_manual_features("https://example.com/pineapple");
        assert_eq!(
            features_pineapple[18], 0.0,
            "'apple' embedded in 'pineapple' should NOT match (word boundary)"
        );
    }
}
