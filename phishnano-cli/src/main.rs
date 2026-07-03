//! # phishnano-cli
//!
//! Command-line interface for phishing URL detection. Supports three modes:
//!
//! 1. **Prediction mode** (default): Score a URL and classify as phishing/normal
//!    ```bash
//!    phishnano-cli "http://suspicious-url.com"
//!    phishnano-cli "http://example.com" --threshold 0.40
//!    phishnano-cli "http://example.com" --model custom_model.bincode
//!    ```
//!
//! 2. **Convert mode** (`--convert`): Convert a JSON model to bincode format
//!    ```bash
//!    phishnano-cli --convert model_data.json model_data.bincode
//!    ```
//!
//! 3. **Help mode** (no arguments): Display usage information
//!
//! ## Classification Threshold
//!
//! The default threshold is 0.45. URLs with a score >= threshold are
//! classified as "Phishing"; otherwise "Normal". Lower the threshold
//! for higher phishing recall (at the cost of more false positives);
//! raise it for fewer false positives (at the cost of missing some phishing).

use anyhow::Context;
use phishnano::{convert_json_to_bincode, load_default_model, load_model_from_path, predict_url};
use std::env;
use std::process;

/// Print usage information to stderr.
fn print_usage() {
    eprintln!("Usage: phishnano-cli <URL> [--threshold <value>]");
    eprintln!("       phishnano-cli --convert <json_path> <bincode_path>");
    eprintln!("");
    eprintln!("Arguments:");
    eprintln!("  <URL>             URL to analyze");
    eprintln!("  --threshold <v>   Classification threshold (default: 0.50)");
    eprintln!("  --convert <json> <bin>  Convert JSON model to bincode format");
    eprintln!("  --model <path>    Load model from file instead of embedded default");
    eprintln!("");
    eprintln!("Classification:");
    eprintln!("  score >= threshold: Phishing");
    eprintln!("  score < threshold:  Normal");
}

/// Entry point for the CLI application.
///
/// Parses command-line arguments and dispatches to the appropriate mode
/// (prediction or conversion). Exits with code 1 on any error.
fn main() {
    let args: Vec<String> = env::args().collect();

    // Require at least one argument (the URL or --convert flag).
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    // --- Convert mode: JSON → bincode ---
    // Usage: phishnano-cli --convert <json_path> <bincode_path>
    if args[1] == "--convert" {
        if args.len() < 4 {
            eprintln!("Error: --convert requires <json_path> and <bincode_path>");
            process::exit(1);
        }
        let json_path = &args[2];
        let bincode_path = &args[3];
        match convert_json_to_bincode(json_path, bincode_path) {
            Ok(size) => {
                println!("Converted: {} -> {}", json_path, bincode_path);
                println!(
                    "Bincode size: {} bytes ({:.2} KB)",
                    size,
                    size as f64 / 1024.0
                );

                // Display compression ratio if the original JSON file exists.
                if let Ok(json_meta) = std::fs::metadata(json_path) {
                    let json_size = json_meta.len();
                    println!(
                        "JSON size:     {} bytes ({:.2} KB)",
                        json_size,
                        json_size as f64 / 1024.0
                    );
                    let ratio = size as f64 / json_size as f64 * 100.0;
                    println!("Compression:   {:.1}% of original", ratio);
                }
            }
            Err(e) => {
                eprintln!("Error: Failed to convert: {}", e);
                process::exit(1);
            }
        }
        return;
    }

    // --- Prediction mode ---
    // The first positional argument is the URL to analyze.
    let url = &args[1];

    // Parse optional flags: --threshold and --model.
    let mut threshold = 0.45;
    let mut model_path: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        if args[i] == "--threshold" && i + 1 < args.len() {
            threshold = args[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("Error: Invalid threshold value");
                process::exit(1);
            });
            i += 2;
        } else if args[i] == "--model" && i + 1 < args.len() {
            model_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }

    // Load the model: use external file if specified, otherwise use the
    // embedded default model compiled into the binary.
    let model = if let Some(path) = model_path {
        load_model_from_path(&path).with_context(|| format!("Failed to load model from {}", path))
    } else {
        load_default_model().context("Failed to load embedded model")
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        process::exit(1);
    });

    // Run inference and display the result.
    let score = predict_url(url, &model);
    let classification = if score >= threshold {
        "Phishing"
    } else {
        "Normal"
    };

    println!("Score: {:.4}", score);
    println!("Classification: {}", classification);
}
