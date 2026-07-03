# phishnano

Lightweight phishing URL detection library with an embedded Random Forest model. Designed for integration into password managers, browser extensions, and security gateways where local, privacy-preserving inference is required.

## Features

- **Embedded model**: Zero configuration, the bincode model is compiled into the library via `include_bytes!`
- **Compact**: ~110 KB embedded model, no external files needed
- **Fast inference**: ~20 microseconds per URL
- **Zero network dependency**: Fully local inference, no API calls
- **Cross-platform**: Windows / macOS / Linux

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

```bash
# Install
cargo install phishnano-cli

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
|-- Cargo.toml                # Library crate + workspace root
|-- src/                      # Library source code
|   |-- lib.rs
|   |-- model.rs              # Model struct, load_default_model(), include_bytes!
|   |-- extractor.rs          # Feature extraction (n-gram + manual)
|   `-- predictor.rs          # Random forest prediction
|-- resources/                # Model files
|   |-- model_data.bincode    # Embedded bincode model (~110 KB)
|   |-- model_data.json       # JSON model for debugging
|   `-- test_features.json    # Cross-language test data
|-- phishnano-cli/            # CLI binary crate (workspace member)
|   |-- Cargo.toml
|   `-- src/
|       `-- main.rs
|-- training/                 # Training scripts (not published to crates.io)
|   |-- data/                 # Training datasets (gitignored)
|   `-- scripts/              # Python training scripts
|       |-- train.py
|       |-- export.py
|       |-- threshold_analysis.py
|       |-- model_analysis.py
|       |-- download_phreshphish.py
|       |-- requirements.txt
|       `-- README.md
|-- .github/
|   `-- workflows/
|       |-- ci.yml            # CI: build + test + clippy + fmt
|       `-- publish.yml       # CD: publish to crates.io on tag push
|-- .gitignore
|-- LICENSE
`-- README.md
```

## Training

The model is pre-trained and embedded in the library. Training data and scripts are maintained in a separate repository and are not included in this crate.

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

- **CI** (`.github/workflows/ci.yml`): Builds and tests on push/PR across Ubuntu, Windows, and macOS. Runs `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo build --release --all`, and `cargo test --all`.
- **Publish** (`.github/workflows/publish.yml`): Publishes both `phishnano` (library) and `phishnano-cli` to crates.io when a version tag (`v*`) is pushed. Requires the `CARGO_REGISTRY_TOKEN` repository secret.

## License

MIT
