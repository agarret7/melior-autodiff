# melior-autodiff

Rust/[Melior](https://github.com/mlir-rs/melior) bindings for [Enzyme](https://github.com/EnzymeAD/Enzyme)'s
MLIR integration: pass in MLIR, run Enzyme as a melior pass, and get back the differentiated MLIR.

Enzyme differentiates at the IR level, as a pass. Name a function and the activity of its arguments
and results, and `enzyme-wrap` generates its derivative as a new function. Or write an
`enzyme.autodiff` (reverse mode) or `enzyme.fwddiff` (forward mode) op where the derivative is needed,
and the `enzyme` pass replaces it with a call to a generated derivative. The usual MLIR lowering
takes it from there.
The C API this crate binds is upstream in Enzyme as of
[EnzymeAD/Enzyme#2859](https://github.com/EnzymeAD/Enzyme/pull/2859).

## Example

Differentiate `square`: pass in MLIR, run the Enzyme passes, and print
the result.

```rust
use melior::{
    ir::Module,
    pass::{transform::create_canonicalizer_pass, PassManager},
};
use melior_autodiff::{
    pass::{self, DerivativeMode},
    utility::create_context,
    Activity,
};

const SOURCE: &str = r#"
module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
}
"#;

let context = create_context();
let mut module = Module::parse(&context, SOURCE).unwrap();

let pm = PassManager::new(&context);
pass::add_differentiate_wrapper_pass(
    &pm,
    "square",
    "dsquare",
    &[Activity::Active],
    &[Activity::Active],
    DerivativeMode::ReverseModeCombined,
)
.unwrap();
pm.add_pass(create_canonicalizer_pass());
pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
pass::add_mathematic_simplification_pass(&pm).unwrap();
pm.run(&mut module).unwrap();

let output = module.as_operation().to_string();
assert!(output.contains("func.func private @dsquare"));
println!("{output}");
```

This prints `@square` along with its derivative, `@dsquare`:

```mlir
func.func private @dsquare(%arg0: f64, %arg1: f64) -> f64 {
  %0 = arith.mulf %arg1, %arg0 fastmath<fast> : f64
  %1 = arith.mulf %arg1, %arg0 fastmath<fast> : f64
  %2 = arith.addf %0, %1 fastmath<fast> : f64     // dr * x + dr * x = 2x dr
  return %2 : f64
}
```

- **The activities** say how Enzyme treats each argument and result of `@square`. `%x` is
  `Active`, so `@dsquare` returns its derivative. The result is `Active` too, so `@dsquare` takes a
  seed, the incoming gradient: `%arg1`, or `dr`.
- **`DerivativeMode`** picks what to generate: `ReverseModeCombined` is ordinary reverse mode, and
  `ForwardMode` generates a function from tangents to tangents.
- **The passes.** `enzyme-wrap` generates `@dsquare` with gradient slots and a tape of cached values.
  `remove-unnecessary-enzyme-ops`, after a canonicalize, removes those where they aren't needed, and
  `enzyme-simplify-math` folds away the arithmetic they leave behind.

From here it's ordinary MLIR, to lower and JIT-compile with melior; `@dsquare` is private, so make it
public first to call it. Where Enzyme ops remain, as caches inside loops do, run
`lower-enzyme-helpers` and then `convert-enzyme-to-memref` first. The [tests](tests/arith.rs) have the
full pipeline.

To differentiate at a particular call site instead, write an `enzyme.autodiff` or `enzyme.fwddiff`
op there (or build one with `autodiff` and `fwddiff`), and run `create_differentiate_pass`, which
replaces each with a call to the derivative it generates.

The other activities: `Dup` passes a shadow alongside the argument, as the next argument. In forward
mode the shadow is the argument's tangent; in reverse mode, for a memref, it's the buffer the gradient
accumulates into. `Const` means no derivative.

## What works

| | Scalars | `scf` control flow | Memrefs | Tensors |
|---|---|---|---|---|
| Reverse mode (`enzyme.autodiff`) | ✓ | | ✓ | ✓ (checked to differentiate, not run) |
| Forward mode (`enzyme.fwddiff`) | ✓ | ✓ | ✓ | ✓, including `width > 1` |
| `linalg.generic`, forward | | | | ✓ (checked to differentiate, not run) |
| `linalg.generic`, reverse | | | ✗ gradient is silently zero | ✗ unsupported by Enzyme |
| `enzyme.batch` | | | | ✓ a scalar function mapped over a tensor (checked to lower, not run) |
| `enzyme.jacobian` | ✗ builds, but nothing lowers it | | | |

Each ✓ is a test in [`tests/`](tests), and the memref `linalg.generic` and `enzyme.jacobian` ✗s
are ignored tests, to turn on once Enzyme handles them. Reverse-mode `linalg.generic` on memrefs
already works on this crate's `impulse` branch, which builds against an Enzyme branch with the fix.

## What's here

| Module | |
|---|---|
| `melior_autodiff` | The Enzyme dialect: `dialect_handle()`, a melior `DialectHandle`; `autodiff`, `fwddiff` and `jacobian`, taking a function name and `&[Activity]`; and `activity_attribute` / `activity_array_attribute` |
| `melior_autodiff::ods` | Every Enzyme op, generated from its TableGen definition the way melior generates upstream dialects: builders, typed ops and accessors |
| `melior_autodiff::pass` | `create_*_pass` for each Enzyme pass with a C API constructor, and `create_lower_enzyme_helpers_pass` for the ops Enzyme emits but doesn't lower to upstream MLIR. `add_differentiate_wrapper_pass` (with `DerivativeMode`) and `add_mathematic_simplification_pass` append `enzyme-wrap` and `enzyme-simplify-math`, which have no C API constructors |
| `melior_autodiff::utility` | `create_context()`, a context with every upstream dialect and Enzyme's; `register_dialect_extensions` for building your own registry; and `register_passes()`, so a textual pipeline can name any Enzyme pass (`enzyme`, `simplify-memref-cache`, ...) |

## Build

Requires LLVM/MLIR 23, and the Enzyme submodule built against it with its MLIR plugin.

The submodule tracks Enzyme's upstream `main` plus one small commit (the `llvm23` branch) that keeps
it building against LLVM 23. Upstream targets LLVM 24, so the commit goes away once `mlir-sys`
supports LLVM 24.

```sh
git clone --recurse-submodules https://github.com/agarret7/melior-autodiff
cd melior-autodiff

# Build Enzyme's MLIR plugin against your LLVM 23 install
cmake -S enzyme/enzyme -B enzyme/build \
  -DLLVM_DIR=$(llvm-config --cmakedir) \
  -DENZYME_MLIR=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build enzyme/build --parallel
```

Then point the build at LLVM 23, in `.cargo/config.toml`:

```toml
[env]
MLIR_SYS_230_PREFIX = "/usr/lib/llvm-23"
TABLEGEN_230_PREFIX = "/usr/lib/llvm-23"
LIBCLANG_PATH = "/usr/lib/llvm-23/lib"
```

`MLIR_SYS_230_PREFIX` and `TABLEGEN_230_PREFIX` are where LLVM 23 is installed (`llvm-config
--prefix`), and `LIBCLANG_PATH` is where bindgen finds its `libclang`. `ENZYME_BUILD_DIR` overrides the
default `enzyme/build` path. Then `cargo test`, and `cargo bench` for the benchmarks.

## Benchmarks

```text
compile/square          ~5.8 ms   (parse → differentiate → lower → JIT compile)
compile/poly_x8         ~8.8 ms
execute/dsquare         ~775 ns   (invoke_packed)
execute_raw/dsquare     ~2 ns     (raw fn ptr via engine.lookup)
```

`invoke_packed` looks the function up by name and calls it through its `void(**)(void**)` wrapper on
every call. For an inner loop, look the function up once with `engine.lookup` and call the pointer
directly, transmuted to its exact C signature.
