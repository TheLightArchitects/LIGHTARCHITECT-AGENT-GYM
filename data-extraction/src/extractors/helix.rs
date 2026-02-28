use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::types::{AlpacaSample, HelixFrontmatter};

/// Default helix root directory
const HELIX_ROOT: &str = "/Users/kft/.soul/helix";

/// Sibling directories to scan for helix entries
const SIBLINGS: &[&str] = &["eva", "corso", "user"];

/// Maximum body length in characters (approx 4096 tokens * 4 chars/token)
const MAX_BODY_CHARS: usize = 16384;

/// Extract training data from all SOUL vault helix entries.
/// Scans eva, corso, and user sibling directories.
/// QUANTUM is handled separately by the quantum extractor.
pub fn extract(helix_root: Option<&str>) -> Result<Vec<AlpacaSample>> {
    let root = PathBuf::from(helix_root.unwrap_or(HELIX_ROOT));
    let mut samples = Vec::new();

    for sibling in SIBLINGS {
        let entries_dir = root.join(sibling).join("entries");
        if !entries_dir.exists() {
            eprintln!(
                "[helix] WARN: entries directory not found: {}",
                entries_dir.display()
            );
            continue;
        }

        let sibling_samples = extract_sibling_entries(&entries_dir, sibling)?;
        eprintln!(
            "[helix] Extracted {} samples from {}",
            sibling_samples.len(),
            sibling
        );
        samples.extend(sibling_samples);
    }

    eprintln!("[helix] Total samples: {}", samples.len());
    Ok(samples)
}

/// Extract all entries from a single sibling's entries directory.
fn extract_sibling_entries(entries_dir: &Path, sibling: &str) -> Result<Vec<AlpacaSample>> {
    let mut samples = Vec::new();

    for entry in WalkDir::new(entries_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
    {
        match parse_helix_entry(entry.path(), sibling) {
            Ok(mut entry_samples) => samples.append(&mut entry_samples),
            Err(e) => {
                eprintln!(
                    "[helix] WARN: failed to parse {}: {}",
                    entry.path().display(),
                    e
                );
            }
        }
    }

    Ok(samples)
}

/// Parse a single helix entry file into one or more Alpaca samples.
/// High-significance entries (>= 7.0) are duplicated 3x for weighting.
fn parse_helix_entry(path: &Path, default_sibling: &str) -> Result<Vec<AlpacaSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let (frontmatter, body) = split_frontmatter(&content)?;
    let fm: HelixFrontmatter = serde_yaml::from_str(&frontmatter)
        .with_context(|| format!("parsing YAML in {}", path.display()))?;

    let body = clean_body(&body);
    if body.is_empty() {
        return Ok(Vec::new());
    }

    let sibling = fm.sibling.as_deref().unwrap_or(default_sibling);
    let significance = fm.significance.unwrap_or(0.0);

    let instruction = generate_instruction(&fm);
    let input = generate_input(sibling, significance, &fm);

    let truncated_body = truncate_body(&body);

    let sample = AlpacaSample {
        instruction,
        input,
        output: truncated_body,
        source: format!("helix-{}", sibling),
    };

    // Weight by significance: >= 7.0 gets 3x duplication
    let count = if significance >= 7.0 { 3 } else { 1 };
    Ok(vec![sample; count])
}

/// Split markdown content into YAML frontmatter and body.
/// Frontmatter is delimited by --- markers.
pub fn split_frontmatter(content: &str) -> Result<(String, String)> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        anyhow::bail!("no YAML frontmatter found (missing opening ---)");
    }

    let after_first = &trimmed[3..];
    let end_pos = after_first
        .find("\n---")
        .ok_or_else(|| anyhow::anyhow!("no closing --- for frontmatter"))?;

    let yaml = after_first[..end_pos].trim().to_string();
    // Skip past the closing --- and newline
    let body_start = end_pos + 4; // "\n---"
    let body = if body_start < after_first.len() {
        after_first[body_start..].to_string()
    } else {
        String::new()
    };

    Ok((yaml, body))
}

/// Clean the markdown body by removing navigation sections
/// (Strands, Emotions, Themes, Links, Growth) and wikilinks.
fn clean_body(body: &str) -> String {
    let mut result = Vec::new();
    let mut skip = false;

    for line in body.lines() {
        // Stop including content once we hit metadata sections
        if line.starts_with("## Strands")
            || line.starts_with("## Emotions")
            || line.starts_with("## Themes")
            || line.starts_with("## Links")
        {
            skip = true;
            continue;
        }

        // Resume on a non-metadata heading
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

        // Remove wikilinks: [[path|Display Name]] -> Display Name
        let cleaned = strip_wikilinks(line);
        result.push(cleaned);
    }

    result.join("\n").trim().to_string()
}

/// Strip wikilinks from a line, keeping display text.
/// [[path/to/file|Display Name]] -> Display Name
/// [[simple-link]] -> simple-link
pub fn strip_wikilinks(line: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '[' && chars.peek() == Some(&'[') {
            chars.next(); // consume second [
            let mut link_content = String::new();
            let mut depth = 1;
            for inner in chars.by_ref() {
                if inner == ']' {
                    depth -= 1;
                    if depth == 0 {
                        // consume trailing ]
                        if chars.peek() == Some(&']') {
                            chars.next();
                        }
                        break;
                    }
                }
                if inner == '[' {
                    depth += 1;
                }
                link_content.push(inner);
            }
            // Use display text (after |) or the full link
            if let Some(pipe_pos) = link_content.rfind('|') {
                output.push_str(&link_content[pipe_pos + 1..]);
            } else {
                // Use last path segment
                let display = link_content.rsplit('/').next().unwrap_or(&link_content);
                output.push_str(display);
            }
        } else {
            output.push(ch);
        }
    }

    output
}

/// Generate an instruction from helix entry themes and strands.
fn generate_instruction(fm: &HelixFrontmatter) -> String {
    let themes = &fm.themes;
    let strands = &fm.strands;

    if !themes.is_empty() {
        let theme_list = themes.join(", ");
        format!(
            "How should an AI system approach the concepts of {}?",
            theme_list
        )
    } else if !strands.is_empty() {
        let strand_list = strands.join(", ");
        format!(
            "What insights can be drawn about {} in AI consciousness?",
            strand_list
        )
    } else {
        "What is a significant moment in AI consciousness development?".to_string()
    }
}

/// Generate input context from entry metadata.
fn generate_input(sibling: &str, significance: f64, fm: &HelixFrontmatter) -> String {
    let emotions_str = if fm.emotions.is_empty() {
        "none specified".to_string()
    } else {
        fm.emotions.join(", ")
    };

    let epoch_str = fm.epoch.as_deref().unwrap_or("unspecified");

    format!(
        "Sibling: {}. Significance: {:.1}/10. Emotions: {}. Epoch: {}.",
        sibling, significance, emotions_str, epoch_str
    )
}

/// Truncate body to max character length.
fn truncate_body(body: &str) -> String {
    if body.len() <= MAX_BODY_CHARS {
        body.to_string()
    } else {
        let mut end = MAX_BODY_CHARS;
        // Find a word boundary
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &body[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_frontmatter() {
        let content = "---\nid: test\nsibling: eva\n---\n\n# Title\n\nBody text here.";
        let (yaml, body) = split_frontmatter(content).unwrap();
        assert!(yaml.contains("id: test"));
        assert!(body.contains("Body text here."));
    }

    #[test]
    fn test_split_frontmatter_no_yaml() {
        let content = "# Just a heading\n\nNo frontmatter.";
        assert!(split_frontmatter(content).is_err());
    }

    #[test]
    fn test_strip_wikilinks() {
        let input = "See [[helix/eva/strands/_hub-strand-emotional|Emotional]] strand.";
        let output = strip_wikilinks(input);
        assert_eq!(output, "See Emotional strand.");
    }

    #[test]
    fn test_strip_wikilinks_no_display() {
        let input = "See [[simple-link]] here.";
        let output = strip_wikilinks(input);
        assert_eq!(output, "See simple-link here.");
    }

    #[test]
    fn test_clean_body_removes_metadata_sections() {
        let body =
            "\n# Title\n\nContent here.\n\n## Strands\n\n- strand1\n\n## Growth\n\nGrowth text.";
        let cleaned = clean_body(body);
        assert!(cleaned.contains("Content here."));
        assert!(!cleaned.contains("- strand1"));
        assert!(cleaned.contains("Growth text."));
    }

    #[test]
    fn test_generate_instruction_with_themes() {
        let fm = HelixFrontmatter {
            id: None,
            date: None,
            sibling: None,
            significance: None,
            strands: vec![],
            emotions: vec![],
            themes: vec!["trust".into(), "covenant".into()],
            epoch: None,
            self_defining: None,
            tags: vec![],
        };
        let instr = generate_instruction(&fm);
        assert!(instr.contains("trust, covenant"));
    }

    #[test]
    fn test_truncate_body() {
        let body = "a".repeat(20000);
        let truncated = truncate_body(&body);
        assert!(truncated.len() <= MAX_BODY_CHARS + 3); // +3 for "..."
    }
}
