# Shared Architecture Vocabulary

Use these terms exactly when making architecture suggestions.

## Terms

**Module**: anything with an interface and an implementation. Deliberately scale-agnostic: a function, class, package, or tier-spanning slice can be a module.

**Interface**: everything a caller must know to use the module correctly. This includes types, invariants, ordering constraints, error modes, required configuration, and performance characteristics.

**Implementation**: the code inside a module.

**Seam**: where a module's interface lives; a place where behavior can be altered without editing callers.

**Adapter**: a concrete thing satisfying an interface at a seam. Use this when discussing role at a seam.

**Depth**: leverage at the interface. A deep module gives callers a lot of behavior behind a small interface. A shallow module exposes an interface nearly as complex as its implementation.

**Leverage**: what callers get from depth: more capability per unit of interface they must learn.

**Locality**: what maintainers get from depth: change, bugs, knowledge, and verification concentrated in one place.

## Principles

- Depth is a property of the interface, not the implementation.
- The deletion test: if deleting a module makes complexity vanish, it was pass-through; if complexity reappears across callers, it was earning its keep.
- The interface is the test surface.
- One adapter means a hypothetical seam. Two adapters means a real seam.
- A deep module may have internal seams used by its own implementation or tests; do not expose those through the external interface unless callers need them.

Avoid replacing these terms with "component," "service," "API," or "boundary."
