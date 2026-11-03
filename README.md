# melior-autodiff

Unofficial Rust/[Melior](https://github.com/mlir-rs/melior) bindings for [Enzyme](https://github.com/EnzymeAD/Enzyme)'s MLIR C API.

Enzyme's MLIR integration exposes automatic differentiation as a dialect pass over MLIR IR. This crate
wraps the C API via bindgen and provides a thin Rust layer for registering Enzyme, running the
differentiation pipeline, and calling JIT-compiled gradients with near-zero overhead.

The C API it binds is upstream in Enzyme as of
[EnzymeAD/Enzyme#2859](https://github.com/EnzymeAD/Enzyme/pull/2859).

## What's here

| | |
|---|---|
| **Dialect setup** | `create_context()`, or for a custom registry `dialect_handle()` (a melior `DialectHandle`) + `register_dialect_extensions` |
| **Op construction** | `autodiff`, `fwddiff`, `jacobian` taking a function name and `&[Activity]`; `activity_attribute` / `activity_array_attribute`; and `ods`, every Enzyme op generated from its TableGen definition the way melior generates upstream dialects (builders, typed ops, accessors) |
| **Passes** | `pass::create_differentiate_pass`, `create_convert_enzyme_to_memref_pass`, `create_batch_pass`, `create_batch_diff_pass`, `create_remove_unnecessary_enzyme_ops_pass`, as melior `Pass`es |
| **Passes by name** | `register_passes()` registers every Enzyme pass, so a textual pipeline can name any of them (`enzyme`, `simplify-memref-cache`, `add-to-op-to-split`, ...) |
| **Lowerings** | `pass::create_lower_enzyme_helpers_pass` (`enzyme.fill_zero`, `enzyme.broadcast`), for ops Enzyme emits but doesn't lower to upstream MLIR |
| **Benchmarks** | Criterion suite measuring compile latency (~6–9 ms), `invoke_packed` overhead (~750 ns), and direct call cost (~2 ns) |

## Quick example

```rust
// Parse a module with an enzyme.autodiff op, lower it, and call the gradient.
let ctx = create_context();
let mut module = Module::parse(&ctx, MLIR_SOURCE).unwrap();

let pm = PassManager::new(&ctx);
pm.add_pass(pass::create_differentiate_pass());
pm.add_pass(pass::create_lower_enzyme_helpers_pass());
pm.add_pass(pass::create_convert_enzyme_to_memref_pass());
pm.add_pass(create_inliner_pass());
pm.add_pass(create_canonicalizer_pass());
pm.add_pass(create_scf_to_control_flow());
pm.add_pass(create_to_llvm());
pm.add_pass(create_reconcile_unrealized_casts());
pm.run(&mut module).unwrap();

let engine = ExecutionEngine::new(&module, 3, &[], false, false);
// The signature must match the compiled function exactly.
let grad: unsafe extern "C" fn(f64, f64) -> f64 =
    unsafe { std::mem::transmute(engine.lookup("dmy_fn")) };
let result = unsafe { grad(3.0, 1.0) }; // 6.0 for d/dx x^2
```

## Build

Requires LLVM/MLIR 23 and the Enzyme submodule built against it.

The submodule tracks Enzyme's upstream `main` plus one small commit (the `llvm23` branch) that keeps
it building against LLVM 23. Upstream targets LLVM 24, so the commit goes away once `mlir-sys`
supports LLVM 24. The `impulse` branch of this crate adds bindings for Enzyme's Impulse dialect on top.

```sh
git clone --recurse-submodules https://github.com/agarret7/melior-autodiff
cd melior-autodiff

# Build Enzyme against your LLVM 23 install
cmake -S enzyme/enzyme -B enzyme/build \
  -DLLVM_DIR=$(llvm-config --cmakedir) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build enzyme/build --parallel

# Point mlir-sys at your LLVM 23 prefix
export MLIR_SYS_230_PREFIX=$(llvm-config --prefix)

cargo test
cargo bench
```

`ENZYME_BUILD_DIR` overrides the default `enzyme/build` path if needed.

## Benchmarks

```
compile/square          ~5.8 ms   (parse → differentiate → lower → JIT compile)
compile/poly_x8         ~8.8 ms
execute/dsquare         ~775 ns   (invoke_packed FFI trampoline)
execute_raw/dsquare     ~2 ns     (raw fn ptr via engine.lookup)
```

`invoke_packed` looks the function up by name and calls it through its `void(**)(void**)` wrapper on
every call. Looking it up once with `engine.lookup` and calling the pointer directly avoids both; do
that for any inner loop (e.g. HMC leapfrog steps).

## Enzyme dialect boundary

`enzyme.jacobian` can be constructed but no Enzyme pass lowers it yet. Its test is marked
`#[ignore]` until upstream adds a lowering for it.
