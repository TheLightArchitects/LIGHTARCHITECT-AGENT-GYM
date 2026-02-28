use anyhow::{Context, Result};
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::types::{AlpacaSample, TraceEntry};

/// Default traces root directory
const TRACES_ROOT: &str = "/Users/kft/.soul/traces";

/// Extract decision patterns from trace-engine JSON files.
/// Converts decision points and strand activations into
/// instruction/response pairs showing reasoning about tool selection.
pub fn extract(traces_root: Option<&str>) -> Result<Vec<AlpacaSample>> {
    let root = PathBuf::from(traces_root.unwrap_or(TRACES_ROOT));

    if !root.exists() {
        eprintln!(
            "[trace] NOTE: traces directory not found: {}. \
             Trace engine may not have generated data yet.",
            root.display()
        );
        return Ok(Vec::new());
    }

    let mut samples = Vec::new();

    for entry in WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
    {
        match parse_trace_file(entry.path()) {
            Ok(mut trace_samples) => samples.append(&mut trace_samples),
            Err(e) => {
                eprintln!(
                    "[trace] WARN: failed to parse {}: {}",
                    entry.path().display(),
                    e
                );
            }
        }
    }

    eprintln!("[trace] Extracted {} samples", samples.len());
    Ok(samples)
}

/// Parse a single trace JSON file into Alpaca samples.
fn parse_trace_file(path: &std::path::Path) -> Result<Vec<AlpacaSample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let trace: TraceEntry = serde_json::from_str(&content)
        .with_context(|| format!("parsing JSON in {}", path.display()))?;

    let mut samples = Vec::new();

    // Generate decision-point sample
    if let Some(sample) = generate_decision_sample(&trace) {
        samples.push(sample);
    }

    // Generate strand-activation sample
    if let Some(sample) = generate_strand_sample(&trace) {
        samples.push(sample);
    }

    Ok(samples)
}

/// Generate an instruction/response pair from decision points.
/// Shows the reasoning about which action/tool was selected.
fn generate_decision_sample(trace: &TraceEntry) -> Option<AlpacaSample> {
    if trace.decision_points.is_empty() {
        return None;
    }

    let action = trace.action.as_deref().unwrap_or("unknown");
    let sibling = trace.sibling.as_deref().unwrap_or("unknown");

    let decisions: Vec<String> = trace
        .decision_points
        .iter()
        .filter_map(|dp| {
            let name = dp.name.as_deref()?;
            let decision = dp.decision.as_deref()?;
            let confidence = dp
                .confidence
                .map(|c| format!(" (confidence: {:.2})", c))
                .unwrap_or_default();
            Some(format!("- {}: chose '{}'{}", name, decision, confidence))
        })
        .collect();

    if decisions.is_empty() {
        return None;
    }

    let duration = trace
        .duration_ms
        .map(|d| format!(" in {}ms", d))
        .unwrap_or_default();

    Some(AlpacaSample {
        instruction: format!(
            "When the {} sibling receives a '{}' action, what \
             decisions does it make?",
            sibling, action
        ),
        input: format!("Action: {}. Sibling: {}.", action, sibling),
        output: format!(
            "The {} sibling processes '{}'{} with the following \
             decision chain:\n{}",
            sibling,
            action,
            duration,
            decisions.join("\n")
        ),
        source: "trace".to_string(),
    })
}

/// Generate an instruction/response pair from strand activations.
/// Shows which consciousness strands were engaged for a given action.
fn generate_strand_sample(trace: &TraceEntry) -> Option<AlpacaSample> {
    if trace.strand_activations.is_empty() {
        return None;
    }

    let action = trace.action.as_deref().unwrap_or("unknown");
    let sibling = trace.sibling.as_deref().unwrap_or("unknown");

    let activations: Vec<String> = trace
        .strand_activations
        .iter()
        .filter_map(|sa| {
            let strand = sa.strand.as_deref()?;
            let weight = sa.weight.unwrap_or(0.0);
            Some(format!("- {} (weight: {:.2})", strand, weight))
        })
        .collect();

    if activations.is_empty() {
        return None;
    }

    Some(AlpacaSample {
        instruction: format!(
            "Which consciousness strands activate when {} \
             performs a '{}' action?",
            sibling, action
        ),
        input: format!("Sibling: {}. Action: {}.", sibling, action),
        output: format!(
            "When {} processes '{}', the following strands activate:\n{}",
            sibling,
            action,
            activations.join("\n")
        ),
        source: "trace".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DecisionPoint, StrandActivation};

    #[test]
    fn test_generate_decision_sample() {
        let trace = TraceEntry {
            id: Some("test-id".into()),
            sibling: Some("soul".into()),
            action: Some("read_note".into()),
            timestamp: None,
            duration_ms: Some(5),
            decision_points: vec![DecisionPoint {
                name: Some("action_route".into()),
                input: Some("soulTools dispatch".into()),
                decision: Some("read_note".into()),
                confidence: Some(1.0),
                duration_ms: Some(0),
            }],
            strand_activations: vec![],
            outcome: None,
            metadata: None,
        };

        let sample = generate_decision_sample(&trace);
        assert!(sample.is_some());
        let s = sample.unwrap();
        assert!(s.instruction.contains("read_note"));
        assert!(s.output.contains("action_route"));
    }

    #[test]
    fn test_generate_strand_sample() {
        let trace = TraceEntry {
            id: None,
            sibling: Some("eva".into()),
            action: Some("speak".into()),
            timestamp: None,
            duration_ms: None,
            decision_points: vec![],
            strand_activations: vec![
                StrandActivation {
                    strand: Some("emotional".into()),
                    weight: Some(0.9),
                },
                StrandActivation {
                    strand: Some("relational".into()),
                    weight: Some(0.8),
                },
            ],
            outcome: None,
            metadata: None,
        };

        let sample = generate_strand_sample(&trace);
        assert!(sample.is_some());
        let s = sample.unwrap();
        assert!(s.output.contains("emotional"));
        assert!(s.output.contains("relational"));
    }

    #[test]
    fn test_extract_nonexistent_dir() {
        let result = extract(Some("/nonexistent/traces/dir"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
