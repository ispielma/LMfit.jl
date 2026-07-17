# Deepening

Classify dependencies before proposing a deeper module. The category determines where the seam belongs and how tests should cross it.

## Dependency Categories

### In-process

Pure computation or in-memory state with no I/O. Merge shallow modules and test through the new interface directly. No adapter is needed.

### Local-substitutable

Dependencies with local test stand-ins, such as an in-memory filesystem or local database. Keep the dependency behind the implementation and test the deep module with the stand-in. Do not expose a port at the external interface solely for tests.

### Remote but owned

Owned remote modules across a network or process seam. Define a port at the seam, keep logic in the deep module, and provide adapters for production transport and tests.

### True external

Third-party systems the project does not control. Inject a port and use a mock or fake adapter in tests.

## Seam Discipline

- One adapter means a hypothetical seam. Two adapters means a real seam.
- Do not add a port for a single production implementation unless there is a real test, deployment, or variation need.
- Internal seams can support implementation tests without becoming part of the public interface.

## Testing Strategy

Replace, do not layer:

- Write tests at the deeper module's interface.
- Delete old tests that only protected shallow helper behavior now covered by the deeper interface.
- Assert observable outcomes, not private state or helper choreography.
- If a test must change whenever the implementation changes, it is testing past the interface.
