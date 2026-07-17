---
name: architecture-deepening
description: Find and design deepening opportunities in a codebase by improving module depth, interface leverage, seam placement, adapter strategy, locality, and testability. Use when the user wants architecture refactoring candidates, consolidation of tightly-coupled modules, better public interfaces, or a more AI-navigable codebase.
metadata:
  uuid: "c1996037-6250-4943-a9d6-ecf3ca899d46"
---

# Architecture Deepening

Use this skill when the task is discovery and design, not immediate patch review. The goal is to find refactors that turn shallow modules into deeper modules with better leverage, locality, and test surfaces.

## Vocabulary

Use these terms consistently:

- **Module**: anything with an interface and an implementation.
- **Interface**: everything a caller must know to use a module correctly, including invariants, ordering, error modes, and configuration.
- **Implementation**: the code inside a module.
- **Seam**: where an interface lives; a place behavior can vary without editing callers.
- **Adapter**: a concrete thing satisfying an interface at a seam.
- **Depth**: leverage at the interface. A deep module hides substantial behavior behind a small interface.
- **Leverage**: capability callers get per unit of interface.
- **Locality**: change, bugs, and verification concentrated in one place.

Prefer these terms over "component," "service," "API," or "boundary." See [references/language.md](references/language.md) when the review depends on exact terminology.

## Explore

Read existing documentation first when present:

- `CONTEXT.md` or context maps
- relevant ADRs in `docs/adr/`
- architecture docs, specs, and repository instructions

Proceed silently if those files do not exist. Do not suggest creating them unless the user asks or the work needs a durable decision record.

Then explore the codebase for friction:

- understanding one concept requires bouncing through many shallow modules
- callers must know nearly as much as the implementation
- tests cross private seams because the public interface is the wrong shape
- pure helpers were extracted for testability but bugs live in orchestration
- tightly coupled modules leak knowledge across their seams
- repeated call-site logic suggests missing leverage
- a seam has only one adapter and no real variation

Apply the deletion test: if deleting a module makes complexity vanish, it was probably pass-through; if deleting it spreads complexity across callers, it may be earning its keep.

## Present Candidates

Present numbered deepening opportunities. For each candidate include:

- **Files**: modules involved.
- **Problem**: why the current interface, seam, or ownership causes friction.
- **Solution**: what would move behind the deeper interface.
- **Benefits**: how leverage, locality, and tests improve.
- **Dependency strategy**: in-process, local-substitutable, remote-owned port plus adapter, or true external dependency.

Use domain language from project docs when available, plus the vocabulary above. Do not propose detailed interfaces yet. Ask the user which candidate to explore.

## Design Interfaces

When the user chooses a candidate, compare at least two materially different interface shapes before recommending one. Use [references/deepening.md](references/deepening.md) for dependency strategy and [references/interface-design.md](references/interface-design.md) for design comparison.

For each design, describe:

1. Interface shape, including invariants and error modes.
2. Example caller usage.
3. Implementation hidden behind the seam.
4. Adapter strategy, if any.
5. Trade-offs in leverage and locality.

Recommend one design or a hybrid. Be explicit about what old tests become obsolete once behavior is tested through the deeper interface.

## Boundaries

- Do not turn discovery into a broad rewrite without user selection.
- Do not introduce a seam unless at least two adapters are justified or the seam is already real.
- Do not preserve a shallow module just because it has tests.
- Do not propose interfaces that only move complexity from implementation into callers.
