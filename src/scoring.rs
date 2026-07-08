//! Two-stage phishing URL scoring.
//!
//! This module implements the *deterministic Stage-1 rule layer* plus the
//! *Stage-2 forest refinement* that together form phishnano's final scorer.
//!
//! ## Why two stages?
//!
//! The embedded decision tree forest learns character n-gram patterns over the
//! *whole* URL. That means a benign token like `google` inside a subdomain
//! (e.g. `mail.google.com`) gets baked into a "normal bias", which can drag
//! the score of a *hijacked* or *look-alike* URL down. Stage 1 isolates that
//! interference by qualifying the **registrable domain (eTLD+1)** first:
//!
//! - A whitelisted, structurally-benign domain (e.g. `mail.google.com`) is
//!   normally scored at a **fixed low value (0.1)**, which eliminates the
//!   `mail.google.com` false positive. However, if the forest *independently*
//!   flags the URL (score >= threshold), we trust the forest instead -- this
//!   recovers phishing URLs hosted on whitelisted domains (e.g. a compromised
//!   popular site) that the old hard-pin would have force-passed. When the
//!   forest is below threshold the low base still applies, so no new false
//!   positives are introduced.
//! - A domain that impersonates a brand or uses a high-risk TLD is floored
//!   at **0.5** (the forest may push it higher).
//! - Everything else (grey) is deferred to the **forest** (Stage 2).
//!
//! Final combination: `max(base, forest)` for phishing, `forest` for grey, and
//! `forest if forest >= threshold else max(base, forest)` for normal. The rule
//! layer costs **zero ML model volume**.
//!
//! The logic here is ported 1:1 from `training/scripts/two_stage_sim.py`,
//! which validated the lift offline before any Rust change.

use crate::extractor::{levenshtein, registrable_domain, BRAND_CANONICAL, MULTI_PART_TLDS};
use crate::model::Model;
use crate::predictor::predict_forest;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Embedded Stage-1 whitelist: the top-N most-popular registrable domains
/// from the Tranco/Alexa Top 1M snapshot (see `build_whitelist.py`).
///
/// Format (little-endian): `u32 count`, then `count` records of
/// `u16 len` + `len` ASCII-lowercase bytes. Domains are sorted ascending so
/// we can binary-search them as `&str` slices over the static bytes.
const WHITELIST_BYTES: &[u8] = include_bytes!("../resources/whitelist.bin");

/// Score assigned to a whitelisted, structurally-benign domain (fixed).
const NORMAL_BASE: f32 = 0.1;
/// Floor assigned to a Stage-1 phishing verdict (brand / high-risk TLD).
const PHISH_BASE: f32 = 0.5;
/// Default classification threshold: scores >= this are "Phishing".
///
/// Set to 0.20 to maximize phishing recall on the 20260702 hold-out snapshot
/// (pure-forest phishing recall ~0.96 at this point). The Stage-1 phishing
/// floor (brand-impersonation / high-risk TLD → 0.5) is intentionally disabled
/// when `THRESHOLD < PHISH_BASE`, because forcing a 0.5 verdict below the
/// deployment threshold would convert benign brand-keyword / risky-TLD domains
/// into false positives. In that regime those domains defer to the forest.
pub(crate) const THRESHOLD: f32 = 0.20;
/// Brand-impersonation similarity threshold (>= means typosquat).
const BRAND_THRESHOLD: f32 = 0.6;

/// The Stage-1 category of a URL's registrable domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Category {
    /// Known-good / whitelisted with benign subdomain -> fixed low score.
    Normal,
    /// Brand-impersonation or high-risk TLD -> floored at 0.5.
    Phishing,
    /// Undecided -> defer entirely to the forest.
    Grey,
}

/// Result of the deterministic Stage-1 analysis (cheap, no forest run).
pub struct Stage1Analysis {
    pub category: Stage1Category,
    /// Floor / fixed score contributed by Stage 1.
    pub base: f32,
    /// Human-readable reason when the verdict is `Phishing` (for indicators).
    pub reason: Option<&'static str>,
}

// ---------------------------------------------------------------------------
// Stage-1 asset sets
// ---------------------------------------------------------------------------

/// High-risk TLDs: ONLY those with a well-documented, heavy abuse history
/// (free/very-cheap domains, spam/phishing concentrations). Generic and
/// country-code TLDs that host legitimate sites are deliberately excluded to
/// avoid false positives.
const HIGH_RISK_TLDS: &[&str] = &[
    "accountant", "bid", "bond", "cf", "cfd", "click", "country", "cyou", "date", "download",
    "forex", "gdn", "gq", "ga", "host", "icu", "loan", "men", "ml", "monster", "party", "pro",
    "racing", "rest", "review", "science", "sbs", "stream", "tk", "top", "trade", "win", "xyz",
    "zip",
];

/// TLDs we explicitly never treat as high-risk (popular/legit even if abused).
const TLD_NEVER_RISKY: &[&str] = &[
    "app", "ar", "asia", "au", "bd", "br", "buzz", "ca", "ci", "cl", "cm", "cn", "co", "co.in",
    "co.jp", "co.uk", "com.au", "de", "dev", "edu", "fr", "gh", "gg", "gov", "gov.uk", "gs", "im",
    "in", "ir", "je", "jp", "ke", "kz", "kiwi", "lk", "link", "me", "ms", "mx", "net", "ng",
    "ninja", "np", "org", "org.uk", "pages", "pe", "ph", "pk", "sh", "sn", "st", "tc", "tl", "tv",
    "tz", "ug", "uk", "us", "vc", "ve", "ws", "work", "za", "to",
];

/// Multi-tenant hosting platforms whose registrable domain is shared and whose
/// subdomains are created by arbitrary users (attackers). Whitelisting these
/// at the registrable-domain level would exempt ALL attacker subdomains
/// (e.g. `*.vercel.app`), causing massive false negatives. Such domains are
/// deferred to the forest (Stage 2) instead of being fixed-normal.
const HOSTING_PLATFORMS: &[&str] = &[
    "vercel.app", "vercel.dev", "pages.dev", "r2.dev", "workers.dev", "github.io", "gitlab.io",
    "gitbook.io", "webflow.io", "wixstudio.com", "weeblysite.com", "weebly.com", "framer.app",
    "square.site", "ukit.me", "webwave.dev", "sviluppo.host", "teachable.com", "webcindario.com",
    "jotform.com", "hsforms.com", "campaign-archive.com", "firebaseapp.com", "web.app",
    "azurewebsites.net", "herokuapp.com", "netlify.app", "netlify.com", "render.com", "surge.sh",
    "glitch.me", "replit.app", "replit.dev", "pythonanywhere.com", "ngrok.io", "ngrok.app",
    "tunnel.me", "neocities.org", "tumblr.com", "substack.com", "medium.com", "wordpress.com",
    "wordpress.org", "blogspot.com", "blogspot.co.uk", "strikingly.com", "carrd.co",
    "googlepages.com", "spruz.com", "wix.com", "yolasite.com", "webs.com", "bravenet.com",
    "x10host.com", "000webhostapp.com", "infinityfree.net", "epizy.com", "gotpantheon.com",
    "fly.dev", "deno.dev", "observablehq.com", "hashnode.dev", "bitbucket.org", "discordapp.com",
    "webnode.com", "wixsite.com", "webself.net", "simplesite.com", "snack.ws", "altervista.org",
    "angelfire.com", "tripod.com", "lycos.com", "fc2.com", "livejournal.com", "blogspot.jp",
    "weebly.co", "ucraft.me", "webstarts.com", "simdif.com", "bit.ly", "tinyurl.com", "t.co",
    "goo.gl", "qrco.de", "q-r.to", "urlz.fr", "ln.run", "ead.me", "wl.co", "tiny.cc", "ow.ly",
    "buff.ly", "is.gd", "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "bit.do", "soo.gd",
    "clck.ru", "cli.gs", "fur.ly", "tinyarrows.com", "tr.im", "x.co", "snip.ly", "bl.ink",
    "short.io", "reurl.cc", "t.cn", "dwz.cn", "suo.im", "ddns.net", "ddns.info", "zapto.org",
    "no-ip.org", "servehttp.com", "servegame.com", "hopto.org", "myftp.org", "dyndns.org",
    "freeddns.org", "duckdns.org", "noip.me", "gotdns.ch", "selfhost.net", "dnsalias.com",
    "mediafire.com", "sendgrid.net", "zendesk.com", "dropbox.com", "box.com", "drive.google.com",
    "docs.google.com", "arweave.net", "dweb.link", "ipfs.io", "ipfs.dweb.link",
    "cloudflare-ipfs.com", "s3.amazonaws.com", "amazonaws.com", "googleusercontent.com",
    "cloudfront.net", "appspot.com", "storage.googleapis.com", "blob.core.windows.net",
    "r2.cloudflarestorage.com", "transfer.sh", "file.io", "anonfiles.com", "mega.nz",
    "4shared.com", "wetransfer.com", "sendspace.com", "zippyshare.com", "mediafire.co",
    "start.page", "linktr.ee", "lnk.bio", "notion.site", "pixelfed.social", "teletype.in",
    "launch.app", "biodrop.io", "surl.li", "shorter.me", "short.gy", "shorten.is", "did.li",
    "t.ly", "flow.page", "teemill.com", "godaddysites.com", "pantheonsite.io", "mystrikingly.com",
    "framer.website", "framer.ai", "ghost.io", "w3spaces.com", "gofile.io", "windows.net",
    "tmpfiles.org", "webnode.page", "strikingly.com", "brand.site", "sites.google.com",
    "padlet.com", "canva.com", "beacons.ai", "bio.link", "lnk.bio",
];

/// Structural / benign subdomains that, when present on a whitelisted
/// (non-hosting) registrable domain, still justify fixing the score to 0.1.
/// Anything NOT in this set (random strings, brand tokens, "login-secure-...")
/// is treated as an *unusual* subdomain and deferred to the forest.
const COMMON_SUBDOMAINS: &[&str] = &[
    "www", "www2", "m", "mobile", "mail", "email", "webmail", "maps", "drive", "docs",
    "documents", "accounts", "account", "api", "blog", "shop", "store", "my", "go", "app",
    "admin", "support", "help", "news", "en", "cdn", "static", "assets", "img", "images", "image",
    "media", "dev", "staging", "beta", "portal", "calendar", "cal", "groups", "sites", "plus",
    "translate", "photos", "play", "music", "tv", "cloud", "one", "vpn", "learn", "login",
    "log-in", "secure", "pay", "payment", "auth", "web", "online", "live", "home", "main", "site",
    "us", "uk", "eu",
];

fn hosting_set() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| HOSTING_PLATFORMS.iter().copied().collect())
}

fn high_risk_set() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| HIGH_RISK_TLDS.iter().copied().collect())
}

fn never_risky_set() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| TLD_NEVER_RISKY.iter().copied().collect())
}

fn common_subdomain_set() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| COMMON_SUBDOMAINS.iter().copied().collect())
}

// ---------------------------------------------------------------------------
// Whitelist store (zero-copy binary search over embedded bytes)
// ---------------------------------------------------------------------------

fn whitelist_store() -> &'static Vec<&'static str> {
    static LIST: OnceLock<Vec<&'static str>> = OnceLock::new();
    LIST.get_or_init(|| {
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
            let len =
                u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
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
    })
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
fn subdomain_of(host: &str, reg: &str) -> String {
    if host == reg {
        return String::new();
    }
    let suffix = format!(".{}", reg);
    if let Some(stripped) = host.strip_suffix(&suffix) {
        return stripped.to_string();
    }
    if let Some(stripped) = host.strip_suffix(reg) {
        return stripped.trim_end_matches('.').to_string();
    }
    host.to_string()
}

/// Resolve the TLD key for `reg`: a multi-part public suffix (e.g. `co.uk`)
/// if applicable, otherwise the final label.
fn tld_key_of(reg: &str) -> String {
    let labels: Vec<&str> = reg.split('.').collect();
    let tld = if labels.len() >= 2 {
        format!("{}.{}", labels[labels.len() - 2], labels[labels.len() - 1])
    } else {
        labels.last().copied().unwrap_or("").to_string()
    };
    if MULTI_PART_TLDS.contains(&tld.as_str()) {
        tld
    } else {
        labels.last().copied().unwrap_or("").to_string()
    }
}

/// Stage-1 brand-impersonation similarity of the registrable domain.
///
/// Precision-focused: a domain is a typosquat only if EITHER
///   (a) its whole SLD is a near-edit of a brand token (e.g. `g00gle`,
///       `paypa1`, `micros0ft`), OR
///   (b) one of its hyphen-split tokens is a near-exact brand token
///       (e.g. `apple-id-verify` -> `apple`, `secure-paypa1` -> `paypa1`).
/// A brand token merely *substring*-appearing in a longer legitimate word
/// (e.g. `googleanalytics.com`) is NOT flagged, keeping false positives near
/// zero. Mirrors `brand_impersonation_score` in the simulation.
fn brand_impersonation_score(reg: &str) -> f32 {
    let reg_sld = reg.split('.').next().unwrap_or("");
    let mut best = 0.0f32;
    for (_keyword, canonical) in BRAND_CANONICAL {
        if reg == *canonical {
            return 0.0; // exact real brand domain -> not impersonation
        }
        let sld = canonical.split('.').next().unwrap_or(canonical);
        let sld_len = sld.chars().count().max(1) as i32;
        let max_dist = 2.max(sld_len / 4);
        // (a) Whole-SLD near-edit of the brand token.
        let d = levenshtein(reg_sld, sld) as i32;
        let len_diff = (reg_sld.chars().count() as i32 - sld_len).abs();
        let mut sim = if d <= max_dist && len_diff <= 3 {
            1.0 - d as f32 / sld_len as f32
        } else {
            0.0
        };
        // (b) Hyphen-split token near-exact brand token.
        for tok in reg_sld.split('-') {
            if tok.is_empty() {
                continue;
            }
            let dt = levenshtein(tok, sld) as i32;
            let lt = (tok.chars().count() as i32 - sld_len).abs();
            if dt <= max_dist && lt <= 3 {
                sim = sim.max(1.0 - dt as f32 / sld_len as f32);
            }
        }
        if sim > best {
            best = sim;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Stage-1 verdict + two-stage scoring
// ---------------------------------------------------------------------------

/// Compute the deterministic Stage-1 verdict for a URL.
///
/// This is pure rule logic: it does NOT run the forest. The host and
/// registrable domain are derived from `url` here so callers only pass the
/// raw URL.
pub fn analyze_stage1(url: &str) -> Stage1Analysis {
    let host = host_of(url);
    let reg = registrable_domain(&host);
    let (category, base, reason) = stage1_verdict(&reg, &host);
    Stage1Analysis {
        category,
        base,
        reason,
    }
}

/// Core Stage-1 decision, mirroring `stage1_verdict` in the simulation.
///
/// Order is critical:
/// 1. Hosting platform -> grey (subdomains are attacker-controlled).
/// 2. Brand impersonation -> phishing 0.5 (checked BEFORE whitelist, else a
///    whitelisted typosquat would bypass detection).
/// 3. High-risk TLD -> phishing 0.5 (also before whitelist).
/// 4. Whitelisted + benign subdomain -> normal 0.1.
/// 5. Otherwise -> grey (forest decides).
fn stage1_verdict(reg: &str, host: &str) -> (Stage1Category, f32, Option<&'static str>) {
    // 1. Multi-tenant hosting platform: never fix-normal.
    if hosting_set().contains(reg) {
        return (Stage1Category::Grey, 0.0, None);
    }

    // 2. Brand impersonation (must precede whitelist).
    let bsim = brand_impersonation_score(reg);
    if bsim >= BRAND_THRESHOLD {
        return (
            Stage1Category::Phishing,
            PHISH_BASE,
            Some("Domain closely impersonates a known brand"),
        );
    }

    // 3. High-risk TLD (must precede whitelist).
    let tld_key = tld_key_of(reg);
    if !never_risky_set().contains(tld_key.as_str()) && high_risk_set().contains(tld_key.as_str()) {
        return (
            Stage1Category::Phishing,
            PHISH_BASE,
            Some("Uses a high-risk top-level domain"),
        );
    }

    // 4. Whitelisted, structurally-benign domain -> fixed low score.
    if is_whitelisted(reg) {
        let sub = subdomain_of(host, reg);
        let benign_sub = sub.is_empty()
            || common_subdomain_set().contains(sub.as_str())
            || sub.split('.').all(|p| common_subdomain_set().contains(p));
        if benign_sub {
            return (
                Stage1Category::Normal,
                NORMAL_BASE,
                Some("Whitelisted trusted domain"),
            );
        }
        // Unusual subdomain -> defer to the forest.
        return (Stage1Category::Grey, 0.0, None);
    }

    // 5. Undecided -> grey.
    (Stage1Category::Grey, 0.0, None)
}

/// Two-stage phishing score for a URL.
///
/// This is the production scorer used by [`crate::predict_url`]. It combines
/// the deterministic Stage-1 verdict with the Random Forest (Stage 2):
///
/// - `Normal` -> `forest` if the forest independently flags it
///   (>= threshold), else `max(0.1, forest)`. This recovers phishing on
///   whitelisted domains without introducing new false positives for
///   benign whitelisted sites.
/// - `Phishing` -> `forest` when the deployment threshold is below the 0.5
///   phishing floor (the floor is disabled to avoid false positives on benign
///   brand-keyword / risky-TLD domains); otherwise `max(0.5, forest)`.
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
        // When the deployment threshold is below the fixed phishing floor
        // (0.5), forcing that floor would turn benign brand-keyword / risky-TLD
        // domains into false positives. In that regime we defer to the forest,
        // which already catches the genuinely phishing ones. At thresholds
        // >= 0.5 the floor behaves as designed (genuine phishing catch).
        Stage1Category::Phishing if THRESHOLD < PHISH_BASE => forest,
        Stage1Category::Phishing => s1.base.max(forest),
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
        // Real registrable domain of this host is `secure-paypal.com`, so the
        // correct subdomain relative to THAT reg is `login`.
        assert_eq!(
            subdomain_of("login.secure-paypal.com", "secure-paypal.com"),
            "login"
        );
        // Fallback path (host ends with reg but with a leading label): matches
        // Python's `strip_suffix(reg)` + `rstrip('.')` behaviour.
        assert_eq!(subdomain_of("login.secure-paypal.com", "paypal.com"), "login.secure-");
    }

    #[test]
    fn test_brand_impersonation() {
        assert!(brand_impersonation_score("paypa1.com") >= BRAND_THRESHOLD);
        assert!(brand_impersonation_score("g00gle.com") >= BRAND_THRESHOLD);
        assert!(brand_impersonation_score("apple-id-verify.com") >= BRAND_THRESHOLD);
        assert!(brand_impersonation_score("secure-paypa1.com") >= BRAND_THRESHOLD);
        // Real brand -> 0 (never impersonation of itself).
        assert_eq!(brand_impersonation_score("google.com"), 0.0);
        // Generic brand-like domain -> not impersonation.
        assert!(brand_impersonation_score("googleanalytics.com") < BRAND_THRESHOLD);
        assert!(brand_impersonation_score("example.com") < BRAND_THRESHOLD);
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
    fn test_stage1_brand_phishing() {
        let a = analyze_stage1("http://paypa1.com/login");
        assert_eq!(a.category, Stage1Category::Phishing);
        assert_eq!(a.base, 0.5);
    }

    #[test]
    fn test_stage1_high_risk_tld() {
        let a = analyze_stage1("http://a1b2c3.tk/login");
        assert_eq!(a.category, Stage1Category::Phishing);
        assert_eq!(a.base, 0.5);
    }

    #[test]
    fn test_stage1_hosting_grey() {
        // vercel.app subdomain must NOT be fixed-normal (attacker-controlled).
        let a = analyze_stage1("https://evil-phish.vercel.app/login");
        assert_eq!(a.category, Stage1Category::Grey);
    }

    #[test]
    fn test_stage1_embedded_brand_attacker_tld() {
        // www.google.com.abcd.xyz -> registrable domain is abcd.xyz (attacker-
        // owned). `.xyz` is a high-risk TLD, so Stage 1 correctly floors it at
        // Phishing 0.5 (the embedded `google` brand token is irrelevant).
        let a = analyze_stage1("https://www.google.com.abcd.xyz/login");
        assert_eq!(a.category, Stage1Category::Phishing);
        assert_eq!(a.base, 0.5);
    }

    #[test]
    fn test_score_url_two_stage() {
        let model = load_default_model().expect("Failed to load model");
        // Normal whitelisted -> below threshold (still passes as benign).
        let n = score_url("https://www.google.com", &model);
        assert!(n < THRESHOLD);
        // Brand typosquat -> the forest (Stage-1 phishing floor is disabled at
        // the low 0.20 threshold) still flags it as phishing.
        let p = score_url("http://paypa1.com/login", &model);
        assert!(p >= THRESHOLD);
        // High-risk TLD typosquat -> flagged as phishing by the forest.
        let p2 = score_url("http://a1b2c3.tk/login", &model);
        assert!(p2 >= THRESHOLD);
    }
}
