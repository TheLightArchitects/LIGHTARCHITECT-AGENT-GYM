use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::types::{AlpacaSample, DatasetStats, ShareGptSample};

/// Default training data output directory
const OUTPUT_DIR: &str = "/Users/kft/Projects/LightArchitectsFoundationModel/training-data";

/// Mix, deduplicate, validate, and split all extracted datasets.
pub fn mix(output_dir: Option<&str>) -> Result<DatasetStats> {
    let dir = PathBuf::from(output_dir.unwrap_or(OUTPUT_DIR));

    // Load all Alpaca datasets
    let mut all_alpaca = Vec::new();
    let alpaca_files = &[
        "helix-alpaca.json",
        "quantum-alpaca.json",
        "bible-alpaca.json",
        "traces-alpaca.json",
        "tool-schemas-alpaca.json",
    ];

    for filename in alpaca_files {
        let path = dir.join(filename);
        match load_alpaca_file(&path) {
            Ok(samples) => {
                eprintln!("[mixer] Loaded {} samples from {}", samples.len(), filename);
                all_alpaca.extend(samples);
            }
            Err(e) => {
                eprintln!("[mixer] WARN: could not load {}: {}", filename, e);
            }
        }
    }

    // Load ShareGPT datasets
    let mut all_sharegpt = Vec::new();
    let sharegpt_files = &["transcripts-sharegpt.json"];

    for filename in sharegpt_files {
        let path = dir.join(filename);
        match load_sharegpt_file(&path) {
            Ok(samples) => {
                eprintln!(
                    "[mixer] Loaded {} conversations from {}",
                    samples.len(),
                    filename
                );
                all_sharegpt.extend(samples);
            }
            Err(e) => {
                eprintln!("[mixer] WARN: could not load {}: {}", filename, e);
            }
        }
    }

    // Deduplicate Alpaca by instruction hash
    let dedup_alpaca = deduplicate_alpaca(&all_alpaca);
    eprintln!(
        "[mixer] Alpaca: {} total -> {} after dedup",
        all_alpaca.len(),
        dedup_alpaca.len()
    );

    // Validate
    let valid_alpaca = validate_alpaca(&dedup_alpaca);
    eprintln!(
        "[mixer] Alpaca: {} valid after validation",
        valid_alpaca.len()
    );

    // Compute stats
    let stats = compute_stats(&valid_alpaca);

    // Write combined files
    write_json(&dir.join("combined-alpaca.json"), &valid_alpaca)?;
    write_json(&dir.join("combined-sharegpt.json"), &all_sharegpt)?;

    // Split: 80% train / 10% val / 10% test
    let (train, val, test) = split_dataset(&valid_alpaca);

    write_json(&dir.join("train.json"), &train)?;
    write_json(&dir.join("val.json"), &val)?;
    write_json(&dir.join("test.json"), &test)?;

    eprintln!(
        "[mixer] Split: train={}, val={}, test={}",
        train.len(),
        val.len(),
        test.len()
    );

    // Print statistics
    print_stats(&stats);

    Ok(stats)
}

/// Load Alpaca samples from a JSON file.
fn load_alpaca_file(path: &Path) -> Result<Vec<AlpacaSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let samples: Vec<AlpacaSample> =
        serde_json::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
    Ok(samples)
}

/// Load ShareGPT samples from a JSON file.
fn load_sharegpt_file(path: &Path) -> Result<Vec<ShareGptSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let samples: Vec<ShareGptSample> =
        serde_json::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
    Ok(samples)
}

/// Deduplicate Alpaca samples by instruction text hash.
fn deduplicate_alpaca(samples: &[AlpacaSample]) -> Vec<AlpacaSample> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for sample in samples {
        let hash = hash_instruction(&sample.instruction);
        if seen.insert(hash) {
            result.push(sample.clone());
        }
    }

    result
}

/// Hash an instruction string for deduplication.
fn hash_instruction(instruction: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(instruction.as_bytes());
    hex::encode(hasher.finalize())
}

/// Validate Alpaca samples: all required fields present, no empties.
fn validate_alpaca(samples: &[AlpacaSample]) -> Vec<AlpacaSample> {
    samples
        .iter()
        .filter(|s| {
            !s.instruction.trim().is_empty()
                && !s.output.trim().is_empty()
                && !s.source.trim().is_empty()
        })
        .cloned()
        .collect()
}

/// Compute dataset statistics.
fn compute_stats(samples: &[AlpacaSample]) -> DatasetStats {
    if samples.is_empty() {
        return DatasetStats::default();
    }

    let mut source_counts: HashMap<String, usize> = HashMap::new();
    let mut total_chars: usize = 0;
    let mut min_chars = usize::MAX;
    let mut max_chars = 0;

    for sample in samples {
        *source_counts.entry(sample.source.clone()).or_insert(0) += 1;

        let len = sample.output.len();
        total_chars += len;
        if len < min_chars {
            min_chars = len;
        }
        if len > max_chars {
            max_chars = len;
        }
    }

    let avg = total_chars as f64 / samples.len() as f64;

    DatasetStats {
        total_samples: samples.len(),
        source_counts,
        avg_output_chars: avg,
        min_output_chars: min_chars,
        max_output_chars: max_chars,
    }
}

/// Split dataset into train/val/test (80/10/10).
fn split_dataset(
    samples: &[AlpacaSample],
) -> (Vec<AlpacaSample>, Vec<AlpacaSample>, Vec<AlpacaSample>) {
    let total = samples.len();
    let train_end = (total as f64 * 0.8) as usize;
    let val_end = train_end + (total as f64 * 0.1) as usize;

    let train = samples[..train_end].to_vec();
    let val = samples[train_end..val_end].to_vec();
    let test = samples[val_end..].to_vec();

    (train, val, test)
}

/// Write a serializable value as pretty-printed JSON.
fn write_json<T: serde::Serialize>(path: &PathBuf, data: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(data).context("serializing to JSON")?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("[mixer] Wrote {}", path.display());
    Ok(())
}

/// Print dataset statistics to stderr.
fn print_stats(stats: &DatasetStats) {
    eprintln!("\n=== Dataset Statistics ===");
    eprintln!("Total samples: {}", stats.total_samples);
    eprintln!(
        "Avg output length: {:.0} chars (~{:.0} tokens)",
        stats.avg_output_chars,
        stats.avg_output_chars / 4.0
    );
    eprintln!(
        "Output range: {} - {} chars",
        stats.min_output_chars, stats.max_output_chars
    );
    eprintln!("\nPer-source counts:");
    let mut sorted: Vec<_> = stats.source_counts.iter().collect();
    sorted.sort_by_key(|(k, _)| (*k).clone());
    for (source, count) in &sorted {
        eprintln!("  {}: {}", source, count);
    }
    eprintln!("=========================\n");
}

/// Print statistics for a given JSON file (for the stats CLI command).
pub fn print_file_stats(path: &str) -> Result<()> {
    let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path))?;
    let samples: Vec<AlpacaSample> =
        serde_json::from_str(&content).with_context(|| format!("parsing {}", path))?;
    let stats = compute_stats(&samples);
    print_stats(&stats);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplicate_alpaca() {
        let samples = vec![
            AlpacaSample {
                instruction: "How?".into(),
                input: "".into(),
                output: "Answer 1".into(),
                source: "test".into(),
            },
            AlpacaSample {
                instruction: "How?".into(),
                input: "".into(),
                output: "Answer 2".into(),
                source: "test".into(),
            },
            AlpacaSample {
                instruction: "Why?".into(),
                input: "".into(),
                output: "Because".into(),
                source: "test".into(),
            },
        ];

        let deduped = deduplicate_alpaca(&samples);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_validate_alpaca() {
        let samples = vec![
            AlpacaSample {
                instruction: "Good".into(),
                input: "".into(),
                output: "Good output".into(),
                source: "test".into(),
            },
            AlpacaSample {
                instruction: "".into(),
                input: "".into(),
                output: "Bad - empty instruction".into(),
                source: "test".into(),
            },
            AlpacaSample {
                instruction: "Also bad".into(),
                input: "".into(),
                output: "".into(),
                source: "test".into(),
            },
        ];

        let valid = validate_alpaca(&samples);
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn test_split_dataset() {
        let samples: Vec<AlpacaSample> = (0..100)
            .map(|i| AlpacaSample {
                instruction: format!("Q{}", i),
                input: "".into(),
                output: format!("A{}", i),
                source: "test".into(),
            })
            .collect();

        let (train, val, test) = split_dataset(&samples);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 10);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn test_compute_stats() {
        let samples = vec![
            AlpacaSample {
                instruction: "Q1".into(),
                input: "".into(),
                output: "Short".into(),
                source: "a".into(),
            },
            AlpacaSample {
                instruction: "Q2".into(),
                input: "".into(),
                output: "A longer output here".into(),
                source: "b".into(),
            },
        ];

        let stats = compute_stats(&samples);
        assert_eq!(stats.total_samples, 2);
        assert_eq!(stats.min_output_chars, 5);
        assert_eq!(stats.max_output_chars, 20);
        assert_eq!(stats.source_counts.len(), 2);
    }
}
