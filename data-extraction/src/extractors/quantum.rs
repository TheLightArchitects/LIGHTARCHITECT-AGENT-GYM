use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::extractors::helix::{self};
use crate::types::{AlpacaSample, HelixFrontmatter};

/// Default QUANTUM entries directory
const QUANTUM_ENTRIES: &str = "/Users/kft/.soul/helix/quantum/entries";

/// Maximum body length in characters
const MAX_BODY_CHARS: usize = 16384;

/// Investigation phases for instruction generation
const INVESTIGATION_PHASES: &[&str] = &[
    "scan (triage)",
    "sweep (evidence collection)",
    "trace (pattern forensics)",
    "probe (multi-source research)",
    "theorize (hypothesis generation)",
    "verify (solution validation)",
    "close (deliverable generation)",
];

/// Extract QUANTUM investigation protocols as Alpaca instruction pairs.
/// QUANTUM entries focus on forensic investigation methodology.
pub fn extract(entries_dir: Option<&str>) -> Result<Vec<AlpacaSample>> {
    let dir = PathBuf::from(entries_dir.unwrap_or(QUANTUM_ENTRIES));

    if !dir.exists() {
        eprintln!(
            "[quantum] WARN: entries directory not found: {}",
            dir.display()
        );
        return Ok(Vec::new());
    }

    let mut samples = Vec::new();

    for entry in WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
    {
        match parse_quantum_entry(entry.path()) {
            Ok(mut entry_samples) => samples.append(&mut entry_samples),
            Err(e) => {
                eprintln!(
                    "[quantum] WARN: failed to parse {}: {}",
                    entry.path().display(),
                    e
                );
            }
        }
    }

    eprintln!("[quantum] Extracted {} samples", samples.len());
    Ok(samples)
}

/// Parse a QUANTUM helix entry into Alpaca samples.
/// Generates investigation-focused instruction/response pairs.
fn parse_quantum_entry(path: &Path) -> Result<Vec<AlpacaSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let (frontmatter_str, body) = helix::split_frontmatter(&content)?;
    let fm: HelixFrontmatter = serde_yaml::from_str(&frontmatter_str)
        .with_context(|| format!("parsing YAML in {}", path.display()))?;

    let body = clean_quantum_body(&body);
    if body.is_empty() {
        return Ok(Vec::new());
    }

    let significance = fm.significance.unwrap_or(0.0);
    let instruction = generate_quantum_instruction(&fm);
    let input = generate_quantum_input(&fm);
    let truncated = truncate(&body);

    let sample = AlpacaSample {
        instruction,
        input,
        output: truncated,
        source: "quantum".to_string(),
    };

    // Weight by significance: >= 7.0 gets 3x duplication
    let count = if significance >= 7.0 { 3 } else { 1 };
    Ok(vec![sample; count])
}

/// Clean QUANTUM entry body, removing navigation metadata.
fn clean_quantum_body(body: &str) -> String {
    let mut result = Vec::new();
    let mut skip = false;

    for line in body.lines() {
        if line.starts_with("## Strands")
            || line.starts_with("## Emotions")
            || line.starts_with("## Themes")
            || line.starts_with("## Links")
        {
            skip = true;
            continue;
        }

        if skip && line.starts_with("## ") {
            let heading = line.trim_start_matches('#').trim();
            if heading != "Strands"
                && heading != "Emotions"
                && heading != "Themes"
                && heading != "Links"
            {
                skip = false;
            } else {
                continue;
            }
        }

        if skip {
            continue;
        }

        // Strip wikilinks
        let cleaned = helix::strip_wikilinks(line);
        result.push(cleaned);
    }

    result.join("\n").trim().to_string()
}

/// Generate investigation-focused instruction from QUANTUM entry.
fn generate_quantum_instruction(fm: &HelixFrontmatter) -> String {
    let themes = &fm.themes;

    if themes.iter().any(|t| t.contains("investigation")) {
        return "Describe the investigation methodology for \
                forensic analysis of security incidents."
            .to_string();
    }

    if themes.iter().any(|t| t.contains("methodology")) {
        return "What systematic approach should be used for \
                multi-phase investigation protocols?"
            .to_string();
    }

    if themes.iter().any(|t| t.contains("evidence")) {
        return "How should evidence be collected, classified, \
                and validated during a forensic investigation?"
            .to_string();
    }

    if !themes.is_empty() {
        let theme_list = themes.join(", ");
        return format!("How should an investigation system handle {}?", theme_list);
    }

    // Default: generate from phases
    let phase_list = INVESTIGATION_PHASES.join(", ");
    format!("Describe the investigation phases: {}", phase_list)
}

/// Generate input context for QUANTUM entries.
fn generate_quantum_input(fm: &HelixFrontmatter) -> String {
    let strands_str = if fm.strands.is_empty() {
        "investigative".to_string()
    } else {
        fm.strands.join(", ")
    };

    let significance = fm.significance.unwrap_or(0.0);
    let epoch_str = fm.epoch.as_deref().unwrap_or("unspecified");

    format!(
        "Sibling: quantum. Strands: {}. Significance: {:.1}/10. Epoch: {}.",
        strands_str, significance, epoch_str
    )
}

/// Truncate text to maximum character length.
fn truncate(text: &str) -> String {
    if text.len() <= MAX_BODY_CHARS {
        text.to_string()
    } else {
        let mut end = MAX_BODY_CHARS;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &text[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_quantum_instruction_investigation() {
        let fm = HelixFrontmatter {
            id: None,
            date: None,
            sibling: Some("quantum".into()),
            significance: Some(8.5),
            strands: vec!["methodical".into()],
            emotions: vec![],
            themes: vec!["investigation".into(), "framework".into()],
            epoch: Some("methodology".into()),
            self_defining: None,
            tags: vec![],
        };
        let instr = generate_quantum_instruction(&fm);
        assert!(instr.contains("investigation methodology"));
    }

    #[test]
    fn test_generate_quantum_instruction_default() {
        let fm = HelixFrontmatter {
            id: None,
            date: None,
            sibling: None,
            significance: None,
            strands: vec![],
            emotions: vec![],
            themes: vec![],
            epoch: None,
            self_defining: None,
            tags: vec![],
        };
        let instr = generate_quantum_instruction(&fm);
        assert!(instr.contains("investigation phases"));
    }
}
