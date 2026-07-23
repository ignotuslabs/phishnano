# phishnano

Lightweight offline phishing URL detection library with an embedded
LightGBM-trained decision tree forest. Designed for integration into
password managers, browser extensions, and security gateways where local,
privacy-preserving inference is required.

## Features

- **Offline & privacy-preserving**: 100% local inference, zero network requests, no data leaves the host
- **Lightweight**: ~123 KB embedded model (bincode format), compiled into the library via `include_bytes!`
- **Fast inference**: ~20 microseconds per URL on commodity hardware
- **Zero configuration**: No runtime files, no API keys, no external services
- **Cross-platform**: Windows / macOS / Linux
- **Bundled CLI**: Binary tool included in the same crate

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
phishnano = "0.1"
```

```rust
use phishnano::{load_default_model, predict_url};

fn main() -> anyhow::Result<()> {
    let model = load_default_model()?;
    let score = predict_url("http://suspicious-url.com", &model);

    if score >= 0.20 {
        println!("Warning: potential phishing site (score={:.4})", score);
    } else {
        println!("Safe (score={:.4})", score);
    }
    Ok(())
}
```

## Detailed Prediction with Risk Indicators

For use cases that need to explain _why_ a URL was flagged (security reports,
tooltips, dashboards), use [`predict_url_detailed`]. It returns the score plus
up to 5 risk indicators. Each indicator carries a stable [`IndicatorGroup`]
enum so consumers can aggregate by risk type without parsing description
strings:

```rust
use phishnano::{load_default_model, predict_url_detailed, IndicatorGroup};
use std::collections::HashMap;

let model = load_default_model()?;
let result = predict_url_detailed("http://192.168.1.1/login", &model);

println!("Score: {:.4}", result.score);
for ind in &result.indicators {
    println!("  [{}] {}", ind.group, ind.description);
}

// `IndicatorGroup` implements Hash + Copy + Eq, so it can be a HashMap key:
let mut counts: HashMap<&IndicatorGroup, usize> = HashMap::new();
for ind in &result.indicators {
    *counts.entry(&ind.group).or_insert(0) += 1;
}

// Or match on a precise enum value, not string parsing
let has_ip = result
    .indicators
    .iter()
    .any(|i| i.group == IndicatorGroup::IpAddress);
```

> `IndicatorGroup`, `IndicatorCategory`, and `IndicatorSource` all implement
> `Copy`, `Hash`, and `Eq`, making them usable as `HashMap`/`HashSet` keys for
> grouping or deduplication. `Indicator` and `Prediction` derive `PartialEq`
> (but not `Eq`/`Hash`, since they contain `f32` weights/scores).
>
> `IndicatorGroup` is `#[non_exhaustive]`: new risk types may be added in
> future versions, so downstream `match` expressions must include a `_` arm.

## Use Cases

- **Password managers**: Warn users before autofilling credentials on suspicious login pages
- **Browser extensions**: Real-time URL classification during navigation
- **Email security gateways**: Scan links in incoming messages without forwarding URLs to cloud APIs
- **Security pipelines**: Batch URL classification in SOAR / SIEM workflows
- **Embedded systems**: On-device phishing detection in network appliances with limited connectivity

## Performance

Measured on a time-split held-out evaluation set (100k URLs, 50k phishing /
50k benign, independent of the training vintage):

| Metric | Value |
|--------|-------|
| Phishing recall | 96.2% |
| Normal recall | 99.5% |
| False positives (per 50k normal) | 255 |
| False negatives (per 50k phishing) | 1886 |
| AUC-ROC | 0.999 |
| Model size (embedded) | ~123 KB (bincode) |
| Inference latency | ~20 us |
| Default threshold | 0.20 |

> The bundled model ships at threshold **0.20** to maximize phishing recall.
> The scorer delegates entirely to the embedded decision tree forest, which
> produces a calibrated phishing probability via
> `sigmoid(init_score + Σ raw_leaf)`.

## CLI Tool

The CLI binary is bundled with the crate. Install it with:

```bash
# Install (includes both library and CLI binary)
cargo install phishnano

# Classify a URL (uses the embedded model, no external files needed)
phishnano-cli "http://example.com"

# With a custom threshold
phishnano-cli "http://example.com" --threshold 0.60

# Load a custom model file instead of the embedded one
phishnano-cli "http://example.com" --model my_model.bincode

# Show risk indicators explaining the score
phishnano-cli "http://example.com" --detailed

# Convert a JSON model to bincode
phishnano-cli --convert model_data.json model_data.bincode
```

## Project Structure

```
phishnano/
|-- Cargo.toml                # Single crate (library + CLI binary)
|-- src/
|   |-- lib.rs                # Library root, re-exports public API
|   |-- model.rs              # Model struct, load_default_model(), include_bytes!
|   |-- extractor.rs          # Feature extraction (n-gram + manual + structural)
|   |-- predictor.rs          # Decision tree traversal and scoring
|   |-- scoring.rs            # Production scorer (delegates to the decision tree forest)
|   `-- bin/
|       `-- phishnano-cli.rs  # CLI binary
|-- resources/                # Model files
|   |-- model_data.bincode    # Embedded bincode model (~123 KB)
|   |-- model_data.json       # JSON model for debugging
|   |-- test_features.json    # Cross-language test data (n-gram)
|   `-- test_struct_features.json # Cross-language test data (structural)
|-- training/                 # Training scripts (gitignored, not published)
|-- .github/
|   `-- workflows/
|       |-- ci.yml            # CI: build + test + clippy + fmt
|       `-- publish.yml       # CD: publish to crates.io on tag push
|-- .gitignore
|-- LICENSE-MIT
|-- LICENSE-APACHE
`-- README.md
```

## Training

The model is pre-trained and embedded in the library. Training data and
scripts are maintained in the `training/` directory (gitignored, not published
to crates.io). To retrain, export a JSON model from `train.py` and run
`phishnano-cli --convert` to produce the bincode that gets embedded.

## Model Architecture

phishnano uses a **LightGBM gradient-boosted decision tree forest** with
**100 trees** (max depth 7), trained on ~500k labeled URLs.

### Feature extraction

Each URL is converted to a **539-dimensional** feature vector:

- **500 n-gram features**: Character 2-grams and 3-grams, hashed into 500
  buckets using MurmurHash3 (unsigned, seed=0)
- **39 manual features**: 21 hand-crafted engineered features plus 18
  structural features. The Rust extractor computes **40** manual features (an
  extra trailing structural feature kept for training-pipeline alignment);
  the embedded model uses the first 39 and ignores the trailing one.

### Scoring: decision tree forest

The scorer delegates entirely to the embedded LightGBM decision tree forest.
The forest produces a calibrated phishing probability in `[0, 1]` via
`sigmoid(init_score + Σ raw_leaf)` — the LightGBM additive scoring semantics.
The default classification threshold is **0.20**.

Final score = `forest`.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): Builds and tests on push/PR across
  Ubuntu, Windows, and macOS. Runs `cargo fmt --check`,
  `cargo clippy -- -D warnings`, `cargo build --release`, and `cargo test`
  (including doctests).
- **Publish** (`.github/workflows/publish.yml`): Publishes `phishnano`
  (library + CLI) to crates.io when a version tag (`v*`) is pushed. Requires
  the `CARGO_REGISTRY_TOKEN` repository secret.

## API Documentation

Full API documentation with examples is available at
**[docs.rs/phishnano](https://docs.rs/phishnano)**.

## License

Dual-licensed under either of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
