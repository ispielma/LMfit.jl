# Repository guidance

## Coding style

- Prefer explicit `for` loops for numerical work and collection traversal. This project follows a readable Fortran/C-influenced style.
- Use `map` only when it provides a clear benefit over a loop. Do not embed a multi-line `do ... end` block directly in `map`; if that idiom is genuinely useful, move the operation into a clearly named helper function.
- Keep numerical types consistent by validating or converting them at the interface where values enter a calculation. Do not defer type reconciliation until allocating an accumulator.
- Once related arrays have a common element type and shape, initialize result arrays from an existing value with patterns such as `zero(f)` rather than rebuilding their type and dimensions manually.
- Prefer straightforward control flow and intermediate variables over compact functional expressions when the latter obscure mutation, ordering, or numerical intent.

## Julia module organization

- Every production Julia source file under `src/` must define exactly one module, and that module must be defined completely in that file.
- Never split one module's namespace across multiple files with `include`. An `include` from a parent module may load a child module, but the included file must contain that child's complete module definition.
- When reorganizing code, either deepen the existing module in its existing file or introduce a new, self-contained module in a new file. Do not create implementation-fragment files that implicitly add definitions to another file's module.
- This one-module-per-file rule is a deliberate namespace guarantee: reading one file must be sufficient to see the complete definition of the module named by that file.

## Tests

- Preserve behavior through public-interface tests.
- Run `julia --project=. -e 'using Pkg; Pkg.test(; coverage=false)'` after changing package behavior.
