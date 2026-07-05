# phishnano

Lightweight offline phishing URL detection library with an embedded Random Forest model. Designed for integration into password managers, browser extensions, and security gateways where local, privacy-preserving inference is required.

## Features

- **Offline & privacy-preserving**: 100% local inference, zero network requests, no data leaves the host
- **Lightweight**: ~110 KB embedded model (bincode format), compiled into the library via `include_bytes!`
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

    if score >= 0.45 {
        println!("Warning: potential phishing site (score={:.4})", score);
    } else {
        println!("Safe (score={:.4})", score);
    }
    Ok(())
}
```

## Use Cases

- **Password managers**: Warn users before autofilling credentials on suspicious login pages
- **Browser extensions**: Real-time URL classification during navigation
- **Email security gateways**: Scan links in incoming messages without forwarding URLs to cloud APIs
- **Security pipelines**: Batch URL classification in SOAR / SIEM workflows
- **Embedded systems**: On-device phishing detection in network appliances with limited connectivity

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Phishing precision | 97% |
| Phishing recall | 93% |
| Normal precision | 93% |
| Normal recall | 97% |
| AUC-ROC | 0.9907 |
| Model size (embedded) | ~110 KB (bincode) |
| Inference latency | ~20 us |
| Default threshold | 0.45 |

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
|   |-- extractor.rs          # Feature extraction (n-gram + manual)
|   |-- predictor.rs          # Random Forest prediction
|   `-- bin/
|       `-- phishnano-cli.rs  # CLI binary
|-- resources/                # Model files
|   |-- model_data.bincode    # Embedded bincode model (~110 KB)
|   |-- model_data.json       # JSON model for debugging
|   `-- test_features.json    # Cross-language test data
|-- training/                 # Training scripts (not published to crates.io)
|   |-- data/                 # Training datasets (gitignored)
|   `-- scripts/              # Python training scripts
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

The model is pre-trained and embedded in the library. Training data and scripts are maintained in the `training/` directory (gitignored, not published to crates.io).

## Model Architecture

phishnano uses a Random Forest classifier with 25 decision trees (max depth 7) trained on 410,000+ labeled URLs.

### Feature extraction

Each URL is converted to a 519-dimensional feature vector:

- **500 n-gram features**: Character 2-grams and 3-grams, hashed into 500 buckets using MurmurHash3 (unsigned, seed=0)
- **19 manual features**: URL length, domain length, special character counts, digit ratio, path/query length, TLD code, sensitive word detection, etc.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 25 |
| max_depth | 7 |
| min_samples_split | 10 |
| class_weight | {0: 1, 1: 2.0} |
| random_state | 42 |
| test_size | 0.2 (stratified) |

### Training data sources

The model is trained on ~496,000 labeled URLs (after deduplication) drawn from public phishing feeds (PhishTank, OpenPhish, Kaggle), legitimate URL rankings (Tranco, Majestic, US federal .gov domains), and the PhreshPhish dataset (HuggingFace).

## CI/CD

- **CI** (`.github/workflows/ci.yml`): Builds and tests on push/PR across Ubuntu, Windows, and macOS. Runs `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo build --release`, and `cargo test`.
- **Publish** (`.github/workflows/publish.yml`): Publishes `phishnano` (library + CLI) to crates.io when a version tag (`v*`) is pushed. Requires the `CARGO_REGISTRY_TOKEN` repository secret.

## API Documentation

Full API documentation with examples is available at **[docs.rs/phishnano](https://docs.rs/phishnano)**.

## License

Dual-licensed under either of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
