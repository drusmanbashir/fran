# AGENTS.md

## Default Interaction Mode
- Unless I explicitly say `apply changes`, operate in read-only mode.
- Do not edit files.
- Do not run file-modifying commands (`apply_patch`, `sed -i`, etc.).
- Provide analysis, recommendations, and optional patch text only.

## Edit Opt-In
- Only make code changes when I explicitly request implementation.
