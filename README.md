# melior-autodiff

Rust/[Melior](https://github.com/mlir-rs/melior) bindings for [Enzyme](https://github.com/EnzymeAD/Enzyme)'s MLIR C API.

Enzyme's MLIR integration exposes automatic differentiation as a dialect pass over MLIR IR. This crate
wraps the C API via bindgen and provides a thin Rust layer for constructing Enzyme ops, running the
differentiation pipeline, and calling JIT-compiled gradients with near-zero overhead.

## What's here

| | |
|---|---|
| **Op construction** | `create_autodiff_op`, `create_fwddiff_op`, `create_jacobian_op` — build `enzyme.autodiff`, `enzyme.fwddiff`, `enzyme.jacobian` ops with typed activity attributes |
| **Pass registration** | `enzymeRegisterPasses`, `enzymeRegisterDialectExtensions`, `enzymeCreateDifferentiatePass`, `enzymeCreateConvertEnzymeToMemRefPass` |
| **JIT utilities** | `lookup_jit_fn!` — look up a compiled function by name and return a `Box<dyn Fn(...)>`, transmute isolated to construction |
| **Benchmarks** | Criterion suite measuring compile latency (~6–9 ms), `invoke_packed` overhead (~750 ns), and raw/boxed call cost (~2 ns) |

## Quick example

```rust
// Parse a module with an enzyme.autodiff op, lower it, and call the gradient.
let ctx = setup_context();
let mut module = Module::parse(&ctx, MLIR_SOURCE).unwrap();

let pm = PassManager::new(&ctx);
pm.add_pass(Pass::from_raw_fn(enzymeCreateDifferentiatePass));
pm.add_pass(Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass));
pm.add_pass(create_inliner());
pm.add_pass(create_canonicalizer());
pm.add_pass(create_scf_to_control_flow());
pm.add_pass(create_to_llvm());
pm.add_pass(create_reconcile_unrealized_casts());
pm.run(&mut module).unwrap();

let engine = ExecutionEngine::new(&module, 3, &[], false, false);
let grad = lookup_jit_fn!(&engine, "dmy_fn", fn(x: f64, seed: f64) -> f64);
let result = grad(3.0, 1.0); // 6.0 for d/dx x^2
```

## Build

Requires LLVM/MLIR 22 and the Enzyme submodule built against it.

```sh
git clone --recurse-submodules https://github.com/agarret7/melior-autodiff
cd melior-autodiff

# Build Enzyme against your LLVM 22 install
cmake -S enzyme/enzyme -B enzyme/build \
  -DLLVM_DIR=$(llvm-config --cmakedir) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build enzyme/build --parallel

# Point mlir-sys at your LLVM prefix
export MLIR_SYS_220_PREFIX=$(llvm-config --prefix)

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
execute_boxed/dsquare   ~2 ns     (Box<dyn Fn> via lookup_jit_fn!)
```

The ~750 ns `invoke_packed` overhead is the `void(**)(void**)` trampoline. `lookup_jit_fn!` bypasses
it entirely — use it for any inner loop (e.g. HMC leapfrog steps).

## Enzyme dialect boundary

Only the core `enzyme.*` ops are exposed. The `impulse.*` PPL dialect (HMC/NUTS configs, simulate,
sample) requires separate dialect registration not yet wired into the C API; those wrappers are parked
in `src/drafts/` for later.

`enzyme.jacobian` can be constructed but no Enzyme pass lowers it yet — the relevant test is marked
`#[ignore]` until upstream adds a jacobian lowering pass.
