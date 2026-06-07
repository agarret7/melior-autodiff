# melior-autodiff

Reverse-mode automatic differentiation as a pass over [Melior](https://github.com/mlir-rs/melior) IR,
with a user-extensible VJP registry. **No Enzyme, no C++, no LLVM/MLIR version lockstep** beyond
whatever Melior itself requires.

This is the scaffold that came out of the design discussion. It is structured so that the AD logic
is decoupled from Melior entirely — the driver, registry, activity model, and rules speak only in
opaque `Val`/`OpId` handles through an emission facade, and **every line that touches Melior lives in
one file** (`src/backend.rs`).

## Why no Enzyme

Enzyme's only extension point for a brand-new primitive op is its C++ autodiff interface. The hard
requirement here — *users add differentiable ops in Rust, without writing C++* — is unreachable on
the Enzyme path. Owning the reverse-mode pass in Rust is the only configuration where a Rust-level
VJP registry is possible. The price is implementing reverse mode for a bounded op set yourself; the
payoff is a clean dependency surface and a registry your users can extend.

## What's real vs. what's a sketch

| Module | Status |
| --- | --- |
| `builder.rs` (facade), `activity.rs`, `cotangent.rs`, `registry.rs`, `reverse.rs` (driver), `rules/`, `testing.rs` | Real, Melior-independent Rust. Compiles and is unit-testable against the symbolic backend. |
| `backend.rs` (Melior impl) | **Sketch.** Every call is marked; verify against your pinned `melior` version. The highest-drift items are op iteration and generic op introspection. |

The four-way dispatch in `reverse.rs` is the heart of the system: **structural → composite →
primitive → opaque-error**. That logic is complete. What remains is real IR plumbing in the backend.

## Prerequisites (for the Melior backend)

- LLVM/MLIR matching Melior's required major version, installed and discoverable
  (`brew install llvm@<N>`; point `MLIR_SYS_<NN>_PREFIX` / `TABLEGEN_<NN>_PREFIX` at it).
- The `melior` version in `Cargo.toml` pinned to your project's.

The logic modules build *without* any of this via the symbolic backend — run those tests in CI even
where LLVM isn't available.

## Adding a differentiable op

Composite (common): build it as a `func.func` from existing differentiable ops. Nothing to register.

New primitive (rare): register one rule. See `register_normal_lpdf` in `src/rules/mod.rs` for a
worked Normal log-density — three operand cotangents, pure Rust, no dialect authoring.

## Next steps, in order

1. Fill in `backend.rs` op iteration + introspection; get `differentiate` running on a
   straight-line `arith`/`math` function end to end.
2. Implement the `scf.if` structural rule, then `scf.for` (the loop "tape" — the hardest piece;
   stash forward intermediates, replay in reverse).
3. Wire Melior's `ExecutionEngine` so a generated gradient is JIT-callable — this is what closes the
   loop for the HMC/VI sampler that consumes it.
4. Build out the `dist.*` primitive rules your PPL needs (log-pdfs, bijector log-dets).

## Layout

```
src/
  lib.rs        public API + Differentiate trait
  builder.rs    Val/OpId handles + AdBuilder emission facade
  activity.rs   Const/Active/Duplicated + reachability activity pass
  cotangent.rs  cotangent accumulation (the chain-rule summation)
  registry.rs   the user-extensible VJP registry
  reverse.rs    the reverse-mode driver (four-way dispatch) — pure logic
  rules/mod.rs  built-in primitive rules + example custom Normal lpdf
  testing.rs    symbolic backend (no LLVM) for unit-testing rules
  backend.rs    THE Melior backend — verify against your pinned version
```