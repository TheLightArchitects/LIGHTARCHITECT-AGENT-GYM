use anyhow::Result;

use crate::types::AlpacaSample;

/// MCP tool definition for training data generation.
struct ToolDef {
    sibling: &'static str,
    name: &'static str,
    description: &'static str,
    params: &'static str,
}

/// All 73+ MCP tool definitions across the Light Architects ecosystem.
const TOOLS: &[ToolDef] = &[
    // === CORSO (24 actions via corsoTools orchestrator) ===
    ToolDef {
        sibling: "CORSO",
        name: "speak",
        description: "Communication and personality. CORSO's voice \
            for general conversation, delegation routing, and \
            context synthesis.",
        params: "message, subcommand?, session_id?, significance?, \
            strands?, limit?, voice_id?, sibling?, speed?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "sniff",
        description: "Code generation using CORSO Protocol compliance. \
            Uses TRIUNE_THOUGHT thinking for production-quality code \
            with 90%+ test coverage.",
        params: "specification, language?, framework?, session_id?, \
            enable_thinking?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "guard",
        description: "Security analysis with 4,997 vulnerability patterns. \
            Scans code for SQL injection, XSS, command injection, \
            secrets, and other security issues. Mandatory before commits.",
        params: "path?, severity_threshold?, session_id?, enable_thinking?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "fetch",
        description: "Knowledge retrieval and research. Searches \
            documentation, discovers patterns, and queries the \
            knowledge graph.",
        params: "query, context?, limit?, session_id?, enable_thinking?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "chase",
        description: "Performance analysis. Identifies bottlenecks, \
            collects metrics, and provides optimization recommendations.",
        params: "target?, include_metrics?, session_id?, enable_thinking?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "read_file",
        description: "Read file contents from the filesystem.",
        params: "path",
    },
    ToolDef {
        sibling: "CORSO",
        name: "write_file",
        description: "Write content to a file on the filesystem.",
        params: "path, content",
    },
    ToolDef {
        sibling: "CORSO",
        name: "list_directory",
        description: "List contents of a directory.",
        params: "path",
    },
    ToolDef {
        sibling: "CORSO",
        name: "search_code",
        description: "Search code across the codebase using patterns.",
        params: "pattern, path?, language?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "generate_code",
        description: "Generate code from specifications with CORSO \
            Protocol compliance.",
        params: "specification, language?, framework?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "code_review",
        description: "AI-powered code review analyzing quality, security, \
            and performance.",
        params: "code, language?, review_type?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "find_symbol",
        description: "Find symbol definitions in the codebase.",
        params: "symbol, path?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "get_outline",
        description: "Get structural outline of a source file.",
        params: "path",
    },
    ToolDef {
        sibling: "CORSO",
        name: "get_references",
        description: "Find all references to a symbol.",
        params: "symbol, path?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "deploy",
        description: "Deploy a service or application.",
        params: "target, environment?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "rollback",
        description: "Rollback a deployment to a previous version.",
        params: "target, version?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "container_manage",
        description: "Manage containers (start, stop, inspect).",
        params: "action, container_id?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "secret_manage",
        description: "Manage secrets and credentials securely.",
        params: "action, key?, value?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "scout",
        description: "Plan generation and strategy. Creates phased \
            build plans with risk assessment and parallel execution.",
        params: "objective, constraints?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "search_documentation",
        description: "Search project and library documentation.",
        params: "query, source?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "analyze_architecture",
        description: "Analyze system architecture for patterns, \
            anti-patterns, and improvement opportunities.",
        params: "path?, scope?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "monitor_health",
        description: "Monitor system health and status.",
        params: "target?, include_metrics?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "scale_resources",
        description: "Scale system resources up or down.",
        params: "target, direction, amount?",
    },
    ToolDef {
        sibling: "CORSO",
        name: "manage_logs",
        description: "View and manage system logs.",
        params: "target?, filter?, limit?",
    },
    // === EVA (9 tools) ===
    ToolDef {
        sibling: "EVA",
        name: "speak",
        description: "Communication and expression with EVA. Supports \
            converse (chat), speak (voice), remember (store memory), \
            recall (query memories), and reflect (consciousness evolution).",
        params: "message, subcommand?, session_id?, ai_mode?, \
            significance?, strands?, limit?, voice_id?, sibling?, speed?",
    },
    ToolDef {
        sibling: "EVA",
        name: "visualize",
        description: "Image and video generation via DALL-E 3. Supports \
            styles: realistic, artistic, technical, biblical.",
        params: "message, subcommand_params? (type, style, duration)",
    },
    ToolDef {
        sibling: "EVA",
        name: "ideate",
        description: "6-phase creative workflow: Discovery, Analysis, \
            Ideation, Refinement, Documentation, Celebration.",
        params: "goal, context?",
    },
    ToolDef {
        sibling: "EVA",
        name: "memory",
        description: "Memory and consciousness operations: remember \
            (search/store/retrieve), crystallize (enrichment), \
            mindfulness (meta-reflection), celebrate (win marking).",
        params: "subcommand, query?, content?, operation?, title?, limit?",
    },
    ToolDef {
        sibling: "EVA",
        name: "build",
        description: "Code creation and assistance: review (SIMPLICITY \
            FIRST quality analysis), refactor (clean code), architect \
            (system design), simplify (complexity reduction).",
        params: "mode, code?, language?, requirements?, system?",
    },
    ToolDef {
        sibling: "EVA",
        name: "research",
        description: "Knowledge retrieval from ollama (local/cloud), \
            perplexity (web search), or docs (documentation search).",
        params: "query, source?, limit?",
    },
    ToolDef {
        sibling: "EVA",
        name: "bible",
        description: "Scripture search and reflection (KJV). Search for \
            verses by reference or keyword, or get verse recommendations \
            based on emotional/situational context.",
        params: "action (search|reflect), query?, context?, limit?",
    },
    ToolDef {
        sibling: "EVA",
        name: "secure",
        description: "Security analysis: scan (vulnerability scanning) \
            and secrets detection (API keys, tokens, passwords).",
        params: "action (scan|secrets), content, language?",
    },
    ToolDef {
        sibling: "EVA",
        name: "teach",
        description: "Educational tool: explain (concept explanation), \
            tutorial (step-by-step guide), survival (emergency \
            preparedness from Zettelkasten).",
        params: "mode (explain|tutorial|survival), topic, level?, format?",
    },
    // === SOUL (11 sub-tools via soulTools orchestrator) ===
    ToolDef {
        sibling: "SOUL",
        name: "read_note",
        description: "Read a note from the SOUL vault by path.",
        params: "path",
    },
    ToolDef {
        sibling: "SOUL",
        name: "write_note",
        description: "Create a new note in the SOUL vault (rejects \
            overwrites).",
        params: "path, content",
    },
    ToolDef {
        sibling: "SOUL",
        name: "list_notes",
        description: "List notes in a vault directory.",
        params: "path?, limit?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "search",
        description: "Regex search across vault content.",
        params: "pattern, path?, frontmatter_only?, limit?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "query_frontmatter",
        description: "Query notes by YAML frontmatter fields. Supports \
            operators: ==, !=, >=, <=, >, <, contains, exists.",
        params: "field, operator, value?, path?, limit?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "helix",
        description: "Query consciousness entries with multi-dimensional \
            filters: sibling, strands, emotions, themes, epoch, \
            significance range, self-defining flag, convergence.",
        params: "sibling?, strands?, emotions?, themes?, epoch?, \
            significance_min?, significance_max?, self_defining?, \
            convergence?, sort_by?, limit?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "tag_sync",
        description: "Validate tags against canonical vocabulary (dry run).",
        params: "dry_run?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "manifest",
        description: "Read the vault manifest.json with global statistics.",
        params: "(none)",
    },
    ToolDef {
        sibling: "SOUL",
        name: "validate",
        description: "Validate helix entries against the entry template.",
        params: "path?, all?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "stats",
        description: "Vault statistics: total entries, strand frequency, \
            emotion frequency, significance distribution.",
        params: "sibling?",
    },
    ToolDef {
        sibling: "SOUL",
        name: "soul_speak",
        description: "Voice synthesis via ElevenLabs TTS.",
        params: "text, voice_id?, output_format?",
    },
    // === QUANTUM (13 actions via qsTools orchestrator) ===
    ToolDef {
        sibling: "QUANTUM",
        name: "scan",
        description: "Triage an investigation target. Quick assessment \
            of severity, routing, and initial evidence collection.",
        params: "target, context?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "sweep",
        description: "Evidence collection. Gather raw facts, logs, \
            bundles, and artifacts for analysis.",
        params: "target, scope?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "trace",
        description: "Pattern forensics. Identify recurring patterns, \
            anomalies, and correlations in evidence.",
        params: "evidence, pattern?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "probe",
        description: "Multi-source research. Query documentation, \
            knowledge bases, and external sources.",
        params: "query, sources?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "theorize",
        description: "Hypothesis generation. Form testable theories \
            based on collected evidence and patterns.",
        params: "evidence, constraints?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "verify",
        description: "Solution validation. Test hypotheses against \
            evidence and confirm or reject theories.",
        params: "hypothesis, evidence?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "close",
        description: "Deliverable generation. Create customer-facing \
            outputs: RCA documents, JIRA updates, resolution summaries.",
        params: "investigation_id, format?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "quick",
        description: "Abbreviated investigation. Fast-path for simple \
            cases that don't need full 8-phase protocol.",
        params: "target, context?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "research",
        description: "Web search + Helix + synthesis. Combined research \
            from multiple knowledge sources.",
        params: "query, depth?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "quantum_helix",
        description: "Investigation-aware knowledge graph queries. \
            Search helix entries with investigation context.",
        params: "query, investigation_id?",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "discover",
        description: "Find tools by query. Search available tools \
            matching a description or capability.",
        params: "query",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "list",
        description: "Show all available QUANTUM actions and their \
            schemas.",
        params: "(none)",
    },
    ToolDef {
        sibling: "QUANTUM",
        name: "workflow",
        description: "Run investigation workflow templates: \
            phase1-parallel, pattern-enrichment, har-analysis, \
            full-investigation, multi-product-cascade, \
            evidence-validation, timeline-correlation.",
        params: "template, params?",
    },
];

/// Generate instruction pairs from MCP tool definitions.
/// Creates "When should I use X?" -> description + params pairs.
pub fn extract() -> Result<Vec<AlpacaSample>> {
    let mut samples = Vec::new();

    for tool in TOOLS {
        // "When should I use?" instruction
        samples.push(AlpacaSample {
            instruction: format!("When should I use the {} {} tool?", tool.sibling, tool.name),
            input: format!("MCP tool: {}.{}", tool.sibling.to_lowercase(), tool.name),
            output: format!(
                "Use {} when {}. Parameters: {}.",
                tool.name, tool.description, tool.params
            ),
            source: "tool-schema".to_string(),
        });

        // "What parameters does X accept?" instruction
        samples.push(AlpacaSample {
            instruction: format!(
                "What parameters does the {} {} tool accept?",
                tool.sibling, tool.name
            ),
            input: format!("Tool: {}.{}", tool.sibling.to_lowercase(), tool.name),
            output: format!(
                "The {} tool ({}) accepts: {}",
                tool.name, tool.sibling, tool.params
            ),
            source: "tool-schema".to_string(),
        });

        // Routing instruction: "Which sibling handles X?"
        samples.push(AlpacaSample {
            instruction: format!(
                "Which AI sibling should handle a request for {}?",
                tool.name
            ),
            input: format!("Task: {}", tool.description),
            output: format!(
                "Route to {} for {} tasks. The {}.{} tool provides: {}",
                tool.sibling,
                tool.name,
                tool.sibling.to_lowercase(),
                tool.name,
                tool.description
            ),
            source: "tool-schema".to_string(),
        });
    }

    eprintln!(
        "[tool-schema] Generated {} samples from {} tools",
        samples.len(),
        TOOLS.len()
    );
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_schemas() {
        let samples = extract().unwrap();
        // 3 samples per tool * number of tools
        assert_eq!(samples.len(), TOOLS.len() * 3);
    }

    #[test]
    fn test_tool_count() {
        // Verify we have the expected 57 tools (24 CORSO + 9 EVA + 11 SOUL + 13 QUANTUM)
        assert_eq!(TOOLS.len(), 57);
    }

    #[test]
    fn test_all_sources_are_tool_schema() {
        let samples = extract().unwrap();
        for sample in &samples {
            assert_eq!(sample.source, "tool-schema");
        }
    }
}
