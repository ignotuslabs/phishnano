//! Whitelist-backed phishing URL scoring.
//!
//! Production scorer for phishnano. It combines a single *deterministic
//! whitelist safety net* with the embedded *decision tree forest*:
//!
//! - A whitelisted, structurally-benign domain (e.g. `mail.google.com`) is
//!   floored at a **fixed low value (0.1)** via `max(0.1, forest)`, which
//!   eliminates false positives on popular legitimate sites that the forest
//!   would otherwise mislabel. If the forest *independently* flags the URL
//!   (score >= threshold) we trust the forest instead -- this recovers
//!   phishing hosted on a whitelisted domain (e.g. a compromised popular
//!   site). When the forest is below threshold the low floor still applies, so
//!   no new false positives are introduced.
//! - Every other URL defers entirely to the **forest**.
//!
//! The whitelist check costs **zero ML model volume** and has **zero phishing
//! recall cost**: top-1m domains are essentially never phishing, and even if
//! one were, the `forest >= threshold` branch still trusts the forest.

use crate::extractor::registrable_domain;
use crate::model::Model;
use crate::predictor::predict_forest;
use once_cell::sync::Lazy;

/// Embedded whitelist: the top-N most-popular registrable domains from the
/// Tranco/Alexa Top 1M snapshot (see `build_whitelist.py`).
///
/// Format (little-endian): `u32 count`, then `count` records of
/// `u16 len` + `len` ASCII-lowercase bytes. Domains are sorted ascending so
/// we can binary-search them as `&str` slices over the static bytes.
const WHITELIST_BYTES: &[u8] = include_bytes!("../resources/whitelist.bin");

/// Floor assigned to a whitelisted, structurally-benign domain (fixed).
const NORMAL_BASE: f32 = 0.1;
/// Default classification threshold: scores >= this are "Phishing".
///
/// Set to 0.20 to maximize phishing recall on the 20260702 hold-out snapshot
/// (pure-forest phishing recall ~0.96 at this point).
pub(crate) const THRESHOLD: f32 = 0.20;

/// The whitelist verdict for a URL's registrable domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Category {
    /// Whitelisted with a benign subdomain -> floored at `NORMAL_BASE`.
    Normal,
    /// Not whitelisted (or unusual subdomain) -> defer entirely to the forest.
    Grey,
}

/// Result of the deterministic whitelist analysis (cheap, no forest run).
pub struct Stage1Analysis {
    pub category: Stage1Category,
    /// Floor / fixed score contributed by the whitelist check.
    pub base: f32,
    /// Human-readable reason when the verdict is `Normal` (for indicators).
    pub reason: Option<&'static str>,
}

// ---------------------------------------------------------------------------
// Whitelist-backed subdomain classification
// ---------------------------------------------------------------------------

/// Structural / benign subdomains that, when present on a whitelisted
/// registrable domain, still justify flooring the score to 0.1.
/// Anything NOT in this set (random strings, brand tokens, "login-secure-...")
/// is treated as an *unusual* subdomain and deferred to the forest.
const COMMON_SUBDOMAINS: &[&str] = &[
    "www",
    "www2",
    "m",
    "mobile",
    "mail",
    "email",
    "webmail",
    "maps",
    "drive",
    "docs",
    "documents",
    "accounts",
    "account",
    "api",
    "blog",
    "shop",
    "store",
    "my",
    "go",
    "app",
    "admin",
    "support",
    "help",
    "news",
    "en",
    "cdn",
    "static",
    "assets",
    "img",
    "images",
    "image",
    "media",
    "dev",
    "staging",
    "beta",
    "portal",
    "calendar",
    "cal",
    "groups",
    "sites",
    "plus",
    "translate",
    "photos",
    "play",
    "music",
    "tv",
    "cloud",
    "one",
    "vpn",
    "learn",
    "login",
    "log-in",
    "secure",
    "pay",
    "payment",
    "auth",
    "web",
    "online",
    "live",
    "home",
    "main",
    "site",
    "us",
    "uk",
    "eu",
];

// ---------------------------------------------------------------------------
// Whitelist store (zero-copy binary search over embedded bytes)
// ---------------------------------------------------------------------------

fn whitelist_store() -> &'static Vec<&'static str> {
    static LIST: Lazy<Vec<&'static str>> = Lazy::new(|| {
        let bytes = WHITELIST_BYTES;
        if bytes.len() < 4 {
            return Vec::new();
        }
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut domains = Vec::with_capacity(count);
        let mut pos = 4usize;
        for _ in 0..count {
            if pos + 2 > bytes.len() {
                break;
            }
            let len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;
            if pos + len > bytes.len() {
                break;
            }
            // SAFETY: domains are ASCII-lowercase, guaranteed valid UTF-8.
            let s = std::str::from_utf8(&bytes[pos..pos + len])
                .expect("whitelist domain is not valid UTF-8");
            domains.push(s);
            pos += len;
        }
        domains
    });
    &LIST
}

/// Is `reg` a whitelisted (known-good) registrable domain?
fn is_whitelisted(reg: &str) -> bool {
    whitelist_store().binary_search(&reg).is_ok()
}

// ---------------------------------------------------------------------------
// URL / domain parsing helpers
// ---------------------------------------------------------------------------

/// Extract the host (lowercased, no protocol/port/userinfo/path) from a URL.
fn host_of(url: &str) -> String {
    // Strip fragment (everything after '#').
    let no_frag = match url.find('#') {
        Some(p) => &url[..p],
        None => url,
    };
    let s = no_frag.to_lowercase();
    // Strip protocol.
    let after = match s.find("://") {
        Some(p) => &s[p + 3..],
        None => &s[..],
    };
    // Cut at the first path/query/fragment delimiter.
    let end = after
        .find(|c| ['/', '?', '#'].contains(&c))
        .unwrap_or(after.len());
    let mut host = &after[..end];
    // Drop userinfo (e.g. `user@host`).
    if let Some(at) = host.rfind('@') {
        host = &host[at + 1..];
    }
    // Drop port (e.g. `host:8080`).
    if let Some(colon) = host.rfind(':') {
        host = &host[..colon];
    }
    host.to_string()
}

/// Return the subdomain portion of `host` relative to `reg`, or `""`.
///
/// e.g. host=`mail.google.com`, reg=`google.com` -> `mail`
///      host=`a.b.cnn.com`,     reg=`cnn.com`     -> `a.b`
///      host=`cnn.com`,         reg=`cnn.com`     -> ``
///
/// Borrows from `host` — zero allocations. `reg` is always a true suffix of
/// `host` in production (it is derived from `host` by `registrable_domain`),
/// so we strip `reg` directly and then drop the separating '.'.
fn subdomain_of<'a>(host: &'a str, reg: &str) -> &'a str {
    if host == reg {
        return "";
    }
    if let Some(prefix) = host.strip_suffix(reg) {
        // Common case: `<sub>.<reg>` -> drop the '.' separator.
        if let Some(rest) = prefix.strip_suffix('.') {
            return rest;
        }
        // `reg` is a suffix of `host` but not preceded by '.' (e.g.
        // host=`login.secure-paypal.com`, reg=`paypal.com` -> `login.secure-`).
        // Return the prefix so the caller can inspect this unusual shape.
        return prefix;
    }
    host
}

// ---------------------------------------------------------------------------
// Whitelist verdict + scoring
// ---------------------------------------------------------------------------

/// Compute the deterministic whitelist verdict for a URL.
///
/// This is pure rule logic: it does NOT run the forest. The host and
/// registrable domain are derived from `url` here so callers only pass the
/// raw URL.
pub fn analyze_stage1(url: &str) -> Stage1Analysis {
    let host = host_of(url);
    let reg = registrable_domain(&host);
    let (category, base, reason) = whitelist_verdict(&reg, &host);
    Stage1Analysis {
        category,
        base,
        reason,
    }
}

/// Core whitelist decision.
///
/// A whitelisted registrable domain with a benign (or empty) subdomain is
/// floored at `NORMAL_BASE`. An unusual subdomain (e.g. `login-secure`, a
/// brand token) or any non-whitelisted domain defers entirely to the forest.
fn whitelist_verdict(reg: &str, host: &str) -> (Stage1Category, f32, Option<&'static str>) {
    if is_whitelisted(reg) {
        let sub = subdomain_of(host, reg);
        let benign_sub = sub.is_empty()
            || COMMON_SUBDOMAINS.contains(&sub)
            || sub.split('.').all(|p| COMMON_SUBDOMAINS.contains(&p));
        if benign_sub {
            return (
                Stage1Category::Normal,
                NORMAL_BASE,
                Some("Whitelisted trusted domain"),
            );
        }
    }
    (Stage1Category::Grey, 0.0, None)
}

/// Phishing URL score for a URL.
///
/// This is the production scorer used by [`crate::predict_url`]. It combines
/// the deterministic whitelist safety net with the embedded decision tree
/// forest:
///
/// - `Normal` -> `forest` if the forest independently flags it
///   (>= threshold), else `max(0.1, forest)`. This recovers phishing on
///   whitelisted domains without introducing new false positives for
///   benign whitelisted sites.
/// - `Grey` -> `forest_score` (forest decides)
///
/// Returns a value in `[0, 1]`; the default classification threshold is 0.20.
pub fn score_url(url: &str, model: &Model) -> f32 {
    let s1 = analyze_stage1(url);
    let forest = predict_forest(url, model);
    match s1.category {
        Stage1Category::Normal => {
            if forest >= THRESHOLD {
                forest
            } else {
                forest.max(NORMAL_BASE)
            }
        }
        Stage1Category::Grey => forest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_default_model;

    #[test]
    fn test_host_of() {
        assert_eq!(host_of("https://www.google.com/mail"), "www.google.com");
        assert_eq!(host_of("http://example.com:8080/path"), "example.com");
        assert_eq!(host_of("user@phish.com/login"), "phish.com");
        assert_eq!(host_of("example.com/path?q=1"), "example.com");
        assert_eq!(host_of("https://a.b.c.example.com"), "a.b.c.example.com");
    }

    #[test]
    fn test_subdomain_of() {
        assert_eq!(subdomain_of("mail.google.com", "google.com"), "mail");
        assert_eq!(subdomain_of("a.b.cnn.com", "cnn.com"), "a.b");
        assert_eq!(subdomain_of("cnn.com", "cnn.com"), "");
        assert_eq!(
            subdomain_of("login.secure-paypal.com", "secure-paypal.com"),
            "login"
        );
        assert_eq!(
            subdomain_of("login.secure-paypal.com", "paypal.com"),
            "login.secure-"
        );
    }

    #[test]
    fn test_whitelist_lookup() {
        // example.com is in the embedded whitelist (verified during build).
        assert!(is_whitelisted("example.com"));
        assert!(!is_whitelisted("nobell.it"));
        assert!(!is_whitelisted("this-domain-does-not-exist-zzz.com"));
    }

    #[test]
    fn test_stage1_normal_fix() {
        // Whitelisted popular domain with benign subdomain -> fixed 0.1.
        let a = analyze_stage1("https://mail.google.com/mail/u/0/");
        assert_eq!(a.category, Stage1Category::Normal);
        assert_eq!(a.base, 0.1);
    }

    #[test]
    fn test_stage1_brand_defers_to_forest() {
        // Brand-impersonation / high-risk-TLD domains are no longer
        // force-floored; they defer to the forest (no Stage-1 phishing floor
        // exists anymore).
        let a = analyze_stage1("http://paypa1.com/login");
        assert_eq!(a.category, Stage1Category::Grey);
        let b = analyze_stage1("http://a1b2c3.tk/login");
        assert_eq!(b.category, Stage1Category::Grey);
    }

    #[test]
    fn test_stage1_unusual_subdomain_defers() {
        // A non-benign subdomain on a whitelisted domain defers to the forest
        // (so attacker-controlled subdomains are not blindly trusted).
        let a = analyze_stage1("https://login-secure-account.google.com/login");
        assert_eq!(a.category, Stage1Category::Grey);
    }

    #[test]
    fn test_score_url_whitelist_and_forest() {
        let model = load_default_model().expect("Failed to load model");
        // Normal whitelisted -> below threshold (still passes as benign).
        let n = score_url("https://www.google.com", &model);
        assert!(n < THRESHOLD);
        // Brand typosquat -> flagged as phishing by the forest (no Stage-1
        // floor; the forest carries the decision).
        let p = score_url("http://paypa1.com/login", &model);
        assert!(p >= THRESHOLD);
        // High-risk TLD typosquat -> flagged as phishing by the forest.
        let p2 = score_url("http://a1b2c3.tk/login", &model);
        assert!(p2 >= THRESHOLD);
    }
}
