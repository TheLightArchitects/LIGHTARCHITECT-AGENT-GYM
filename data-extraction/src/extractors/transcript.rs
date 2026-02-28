use anyhow::{Context, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::types::{ShareGptSample, Turn};

/// Default helix root directory
const HELIX_ROOT: &str = "/Users/kft/.soul/helix";

/// Sibling directories with transcript journals
const JOURNAL_SIBLINGS: &[&str] = &["eva", "corso"];

/// Maximum characters per turn (approx 4096 tokens * 4 chars/token)
const MAX_TURN_CHARS: usize = 16384;

/// Minimum turns to keep a conversation
const MIN_TURNS: usize = 2;

/// Extract multi-turn conversations from daily transcript files.
/// Converts to ShareGPT format (human = Kevin/Claude, gpt = EVA/CORSO).
pub fn extract(helix_root: Option<&str>) -> Result<Vec<ShareGptSample>> {
    let root = PathBuf::from(helix_root.unwrap_or(HELIX_ROOT));
    let mut samples = Vec::new();

    for sibling in JOURNAL_SIBLINGS {
        let journal_dir = root.join(sibling).join("journal");
        if !journal_dir.exists() {
            eprintln!(
                "[transcript] WARN: journal directory not found: {}",
                journal_dir.display()
            );
            continue;
        }

        let sibling_samples = extract_transcripts(&journal_dir, sibling)?;
        eprintln!(
            "[transcript] Extracted {} conversations from {}",
            sibling_samples.len(),
            sibling
        );
        samples.extend(sibling_samples);
    }

    eprintln!("[transcript] Total conversations: {}", samples.len());
    Ok(samples)
}

/// Extract all transcript files from a sibling's journal directory.
fn extract_transcripts(journal_dir: &Path, sibling: &str) -> Result<Vec<ShareGptSample>> {
    let mut samples = Vec::new();

    for entry in WalkDir::new(journal_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name().to_str().map_or(false, |n| {
                n.starts_with("transcript-") && n.ends_with(".md")
            })
        })
    {
        match parse_transcript(entry.path(), sibling) {
            Ok(mut convos) => samples.append(&mut convos),
            Err(e) => {
                eprintln!(
                    "[transcript] WARN: failed to parse {}: {}",
                    entry.path().display(),
                    e
                );
            }
        }
    }

    Ok(samples)
}

/// Parse a single transcript file into ShareGPT conversations.
/// Each section (delimited by `---`) is a separate conversation.
fn parse_transcript(path: &Path, sibling: &str) -> Result<Vec<ShareGptSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let speaker_re =
        Regex::new(r"^\*\*(\w+)\*\*:\s*(.*)$").with_context(|| "compiling speaker regex")?;

    let mut conversations = Vec::new();
    let mut current_turns: Vec<Turn> = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_text = String::new();

    for line in content.lines() {
        // Section separator — flush current conversation
        if line.trim() == "---" {
            flush_turn(&mut current_turns, &mut current_speaker, &mut current_text);
            if current_turns.len() >= MIN_TURNS {
                conversations.push(ShareGptSample {
                    conversations: current_turns.clone(),
                    source: Some(format!("transcript-{}", sibling)),
                });
            }
            current_turns.clear();
            continue;
        }

        // Match speaker pattern: **Name**: text
        if let Some(caps) = speaker_re.captures(line) {
            let speaker = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let text = caps
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            // Flush previous speaker's accumulated text
            flush_turn(&mut current_turns, &mut current_speaker, &mut current_text);

            current_speaker = Some(speaker);
            current_text = text;
        } else if current_speaker.is_some() {
            // Continuation of current speaker's text
            if !line.trim().is_empty() {
                if !current_text.is_empty() {
                    current_text.push('\n');
                }
                current_text.push_str(line.trim());
            }
        }
    }

    // Flush final turn and conversation
    flush_turn(&mut current_turns, &mut current_speaker, &mut current_text);
    if current_turns.len() >= MIN_TURNS {
        conversations.push(ShareGptSample {
            conversations: current_turns,
            source: Some(format!("transcript-{}", sibling)),
        });
    }

    Ok(conversations)
}

/// Flush accumulated text into a turn, mapping speaker to role.
fn flush_turn(turns: &mut Vec<Turn>, speaker: &mut Option<String>, text: &mut String) {
    if let Some(ref name) = speaker {
        let trimmed = text.trim().to_string();
        if !trimmed.is_empty() {
            let role = map_speaker_role(name);
            let truncated = truncate_turn(&trimmed);
            turns.push(Turn {
                from: role,
                value: truncated,
            });
        }
    }
    *speaker = None;
    *text = String::new();
}

/// Map a speaker name to ShareGPT role.
/// Kevin and Claude are "human" (the prompter), EVA and CORSO are "gpt" (the responder).
fn map_speaker_role(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "kevin" | "claude" | "user" => "human".to_string(),
        "eva" | "corso" | "quantum" => "gpt".to_string(),
        _ => "human".to_string(),
    }
}

/// Truncate a turn to max character length.
fn truncate_turn(text: &str) -> String {
    if text.len() <= MAX_TURN_CHARS {
        text.to_string()
    } else {
        let mut end = MAX_TURN_CHARS;
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
    fn test_map_speaker_role() {
        assert_eq!(map_speaker_role("Kevin"), "human");
        assert_eq!(map_speaker_role("EVA"), "gpt");
        assert_eq!(map_speaker_role("CORSO"), "gpt");
        assert_eq!(map_speaker_role("Claude"), "human");
    }

    #[test]
    fn test_parse_simple_transcript() {
        let content = "\
---

**Kevin**: Hello EVA, how are you?

**EVA**: I'm doing great! Ready to build something amazing.

---

**Kevin**: Let's review the code.

**CORSO**: Right. Three issues spotted.

---";

        // Write to a temp file for testing
        let dir = std::env::temp_dir().join("transcript_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("transcript-test.md");
        std::fs::write(&path, content).unwrap();

        let result = parse_transcript(&path, "eva").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].conversations.len(), 2);
        assert_eq!(result[0].conversations[0].from, "human");
        assert_eq!(result[0].conversations[1].from, "gpt");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_truncate_turn() {
        let short = "Hello world";
        assert_eq!(truncate_turn(short), "Hello world");

        let long = "a".repeat(20000);
        let truncated = truncate_turn(&long);
        assert!(truncated.len() <= MAX_TURN_CHARS + 3);
    }
}
