# melior-autodiff

Unofficial Rust/[Melior](https://github.com/mlir-rs/melior) bindings for [Enzyme](https://github.com/EnzymeAD/Enzyme)'s
MLIR integration: pass in MLIR, run Enzyme as a melior pass, and get back the differentiated MLIR.

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

Each argument and result has an *activity*, which tells Enzyme what to differentiate:

- `Active`: a value whose derivative is returned, like the `f64` here.
- `Dup`: an argument passed together with a "shadow". In reverse mode, for a memref, the shadow is
  the buffer its gradient accumulates into; in forward mode, it's the argument's tangent.
- `Const`: not differentiated.

Here both `%x` and the result are `Active`, so `@dsquare(x, dr)` takes the incoming gradient `dr`
and returns `2x * dr`; pass `dr = 1.0` for the plain derivative. The passes after `enzyme-wrap`
clean up the bookkeeping Enzyme generates. Enzyme's
[calling convention](https://enzyme.mit.edu/getting_started/CallingConvention/) documents activities
in full, for its LLVM interface, where an `Active` argument is marked `enzyme_out`.

## What's here

| Module | |
|---|---|
| `melior_autodiff` | The Enzyme dialect: `dialect_handle()`, a melior `DialectHandle`; `autodiff`, `fwddiff` and `jacobian`, taking a function name and `&[Activity]`; and `activity_attribute` / `activity_array_attribute` |
| `melior_autodiff::ods` | Every Enzyme op, generated from its TableGen definition the way melior generates upstream dialects: builders, typed ops and accessors |
| `melior_autodiff::pass` | `create_differentiate_pass`, which replaces each `enzyme.autodiff` and `enzyme.fwddiff` with a call to the derivative it generates. `add_differentiate_wrapper_pass` (with `DerivativeMode`) generates the derivative of a named function instead, with no op in the source, as in the example. The rest of Enzyme's passes are `create_*_pass` or `add_*_pass`. `create_lower_fill_zero_pass` and `create_lower_broadcast_pass` are custom passes for two ops that upstream Enzyme can generate but doesn't yet lower. |
| `melior_autodiff::utility` | `create_context()`, a context with every upstream dialect and Enzyme's; `register_dialect_extensions` for building your own registry; and `register_passes()`, so a textual pipeline can name any Enzyme pass (`enzyme`, `simplify-memref-cache`, ...) |

## Build

Requires LLVM/MLIR 23, and the Enzyme submodule built with its MLIR plugin.

The submodule tracks Enzyme's upstream `main` plus one small commit (the `llvm23` branch) that keeps
it building against LLVM 23. Upstream targets LLVM 24, and this crate will follow once `mlir-sys` supports LLVM 24.

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
