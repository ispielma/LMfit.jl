# Interface Design

Use this process when a user chooses a deepening candidate and wants possible interface shapes.

## Frame The Problem

Start by naming:

- constraints the new interface must satisfy
- invariants and error modes callers should not have to rediscover
- dependency category from `deepening.md`
- what implementation complexity should move behind the seam
- what existing tests should migrate to the new interface

Use a short code sketch only to ground constraints. Do not present it as the preferred design yet.

## Compare Designs

Produce at least two materially different designs, and use three when the choice is high-impact:

- minimal interface with 1-3 entry points
- flexible interface optimized for extension
- common-case interface optimized for the dominant caller
- port-and-adapter design when a remote-owned or true external dependency requires it

For each design include:

1. Interface: methods, parameters, invariants, ordering rules, and error modes.
2. Usage example.
3. Implementation hidden behind the seam.
4. Dependency and adapter strategy.
5. Trade-offs in depth, leverage, and locality.

## Recommend

Compare the designs in prose. Recommend the strongest design or a hybrid, and explain what complexity moves out of callers and what tests become durable through the new interface.
