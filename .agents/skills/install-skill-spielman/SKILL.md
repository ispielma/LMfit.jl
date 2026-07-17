---
name: install-skill-spielman
description: Push skills from the AgentSkillsSpielman source-of-truth repository into destination projects. Use when the user asks to list available Spielman skills or skillsets, validate the source repository, install copied skills into a project, install a skillset, refresh copied project skills by matching metadata.uuid, or explicitly create development symlinks with --link.
metadata:
  uuid: "3116160d-3036-4c6f-b2d4-5cf5a4e28f35"
---

# Install Skill Spielman

Use this skill for source-to-project workflows: AgentSkillsSpielman -> project `.agents/skills` and declared shared protocols -> project `.agents/protocols`.

For the opposite direction, where a project/global/external skill should be imported or used to update AgentSkillsSpielman, use `grab-skill`.

## No-Context Invocation

If the user invokes this skill with no added context, respond with this menu and stop:

```text
What source-to-project action do you want me to do?
1. List skills
2. List skillsets
3. Validate the source repository
4. Install one or more skills
5. Install one or more skillsets
6. Refresh copied skills in a project
```

## Operating Model

Install skills by copying them into the target project's `.agents/skills` directory by default, so destination repositories can review, diff, commit, and roll back skill contents normally.

When a source skill declares protocol dependencies in `agents/spielman.yaml`, the manager also copies those files from `protocols/` into the target project's `.agents/protocols/` directory.

Use `refresh` to copy newer source-of-truth skill contents into a project only when `metadata.uuid` matches. Use `--link` only when the user explicitly wants a live symlink for development.

## Commands

Use this skill's bundled `scripts/manage-skills` for all work. Resolve the skill directory first and run the script by path; do not run bare `scripts/manage-skills` from an arbitrary project working directory unless you are already at the AgentSkillsSpielman repository root.

The installed copy uses `agents/spielman.yaml` to find the AgentSkillsSpielman source repository. In this skill, use only these modes:

- `list`
- `list-sets`
- `validate`
- `install`
- `install-set`
- `refresh`

List available skills:

```bash
<install-skill-spielman>/scripts/manage-skills list
```

List available skillsets:

```bash
<install-skill-spielman>/scripts/manage-skills list-sets
```

Validate this repository's skills:

```bash
<install-skill-spielman>/scripts/manage-skills validate
```

Install one or more skills into the current project:

```bash
<install-skill-spielman>/scripts/manage-skills install tdd test-cleanup
```

Install one or more skills into a specific project:

```bash
<install-skill-spielman>/scripts/manage-skills install --project /path/to/project tdd test-cleanup
```

Install one or more skillsets into the current project, replacing existing copied skills only if `--overwrite` is passed:

```bash
<install-skill-spielman>/scripts/manage-skills install-set labscript
```

Refresh copied project skills by UUID:

```bash
<install-skill-spielman>/scripts/manage-skills refresh --project /path/to/project
<install-skill-spielman>/scripts/manage-skills refresh --project /path/to/project tdd test-cleanup
```

Create development symlinks instead of copies only when explicitly requested:

```bash
<install-skill-spielman>/scripts/manage-skills install --link --project /path/to/project tdd
```

Do not use `grab` or `update` from this skill. Those modes pull changes into AgentSkillsSpielman and belong to `grab-skill`.

## Resolution Rules

For skills, accept either a skill name or a repository-relative skill path:

```text
tdd
skills/engineering/tdd
```

For skillsets, accept either a skillset name or a repository-relative skillset path:

```text
labscript
skillsets/labscript.yaml
```

If a name is ambiguous, stop and ask the user which path they want.

## Rules

1. Resolve requested skills or skillsets against the AgentSkillsSpielman repository.
2. Create the target directory if needed:

```text
<project>/.agents/skills/
```

3. Copy one skill directory per skill:

```text
<project>/.agents/skills/<skill-name>/
```

4. Copy each declared protocol dependency:

```text
<project>/.agents/protocols/<domain>/<file>
```

5. If the target entry already exists, do not overwrite it unless the user explicitly requested `--overwrite`.
6. Use `refresh` rather than `install --overwrite` when the user wants to update already-copied skills. Refresh requires matching `metadata.uuid`.
7. After installation, tell the user that Codex should detect the new skills automatically, but a restart or force reload may be needed if they do not appear.
8. If the user wants to preserve edits made inside a destination project, do not refresh over them. Use `grab-skill` to pull those edits back into AgentSkillsSpielman first.

## Skillsets

Skillsets live in `skillsets/*.yaml` at the AgentSkillsSpielman repository root. They are convenience bundles for the installer; Codex does not read them natively.

The scripts support the simple repository-local format:

```yaml
name: labscript
description: Skills commonly used while working on the labscript suite.
skills:
  - skills/engineering/narrow-patch
  - skills/engineering/code-quality-review
```
