# melior-autodiff

[![CI](https://github.com/agarret7/melior-autodiff/actions/workflows/ci.yml/badge.svg)](https://github.com/agarret7/melior-autodiff/actions/workflows/ci.yml)

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
    Some("dsquare"),
    &[Activity::Active],
    &[Activity::Active],
    DerivativeMode::ReverseModeCombined,
)
.unwrap();
pm.add_pass(create_canonicalizer_pass());
pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
pm.add_pass(pass::create_mathematic_simplification_pass());
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
simplify the generated program. Enzyme's
[calling convention](https://enzyme.mit.edu/getting_started/CallingConvention/) documents activities
in full, for its LLVM interface, where an `Active` argument is marked `enzyme_out`.

The [`examples/`](examples) are a good place to start; each JIT-runs a derivative and checks it.
`activities` shows how each activity variant shapes a derivative's signature, `gradients` builds
gradient programs from Rust, `jvp` and `vjp` apply forward and reverse mode to arrays, and
`expression` compiles a Rust expression tree to MLIR at runtime and differentiates it. Run one with
`cargo run --example <name>`.

## What's here

| Module | |
|---|---|
| `melior_autodiff` | The Enzyme dialect: `dialect_handle()`, a melior `DialectHandle`; `autodiff` and `fwddiff`, taking a function name and `&[Activity]`; `batch`, taking a function name and a batch shape; and `activity_attribute` / `activity_array_attribute` |
| `melior_autodiff::ods` | Every Enzyme op, generated from its TableGen definition the way melior generates upstream dialects: builders, typed ops and accessors. It follows Enzyme's op definitions, so it changes when they do. |
| `melior_autodiff::pass` | `add_enzyme_lowering`, which generates the derivatives and lowers Enzyme's ops to upstream MLIR, leaving the rest of the pipeline to you. Reverse mode through `cf` block arguments currently gives wrong gradients; the ignored tests list Enzyme's other known gaps. `create_differentiate_pass`, which replaces each `enzyme.autodiff` and `enzyme.fwddiff` with a call to the derivative it generates. `add_differentiate_wrapper_pass` (with `DerivativeMode`) generates the derivative of a named function instead. The rest of Enzyme's passes are `create_*_pass`, one per pass, with its default options. Enzyme has no pass yet for `enzyme.broadcast`, so `add_enzyme_lowering` lowers it with a pass internal to this crate. |
| `melior_autodiff::utility` | `create_context()`, a context with every upstream dialect and Enzyme's; `register_dialect_extensions` for building your own registry; and `register_passes()`, so a textual pipeline can name any Enzyme pass (`enzyme`, `simplify-memref-cache`, ...) |

## Build

Requires LLVM/MLIR 23, CMake, and Rust from [rustup](https://rustup.rs), which picks up the
toolchain pinned in `rust-toolchain.toml`. On Debian or Ubuntu, LLVM and MLIR 23 come from
[apt.llvm.org](https://apt.llvm.org), along with the libraries the static LLVM build links:

```sh
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 23
sudo apt-get install libmlir-23-dev mlir-23-tools libclang-23-dev libpolly-23-dev \
  zlib1g-dev libzstd-dev libxml2-dev libz3-dev cmake
```

Enzyme is a submodule, and the build script builds the part this crate needs, Enzyme's MLIR C API,
against your LLVM 23 the first time you build (a few minutes). The submodule tracks Enzyme's upstream
`main` plus one small commit (the `llvm23` branch) that keeps it building against LLVM 23. Upstream
targets LLVM 24, and this crate will follow once `mlir-sys` supports LLVM 24.

```sh
git clone --recurse-submodules https://github.com/agarret7/melior-autodiff
cd melior-autodiff
```

Point the build at LLVM 23, in `.cargo/config.toml`:

```toml
[env]
MLIR_SYS_230_PREFIX = "/usr/lib/llvm-23"
TABLEGEN_230_PREFIX = "/usr/lib/llvm-23"
LIBCLANG_PATH = "/usr/lib/llvm-23/lib"
```

`MLIR_SYS_230_PREFIX` and `TABLEGEN_230_PREFIX` are where LLVM 23 is installed (`llvm-config
--prefix`), and `LIBCLANG_PATH` is where bindgen finds its `libclang`. Then `cargo test`, and `cargo
bench` for the benchmarks.

Enzyme is compiled with the clang in `MLIR_SYS_230_PREFIX` unless `CC` and `CXX` say otherwise. Each
compile job can take a couple of gigabytes, so the build runs at most 4 (fewer if Cargo runs fewer);
set `ENZYME_BUILD_JOBS` to change that.

To use an Enzyme you build yourself instead, for example to work on Enzyme, set `ENZYME_BUILD_DIR`
to its build directory. It needs only the `MLIRCAPIEnzyme` and `MLIRLLVMExt` targets:

```sh
cmake -S enzyme/enzyme -B enzyme/build -DLLVM_DIR=$(llvm-config --cmakedir) -DENZYME_MLIR=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build enzyme/build --target MLIRCAPIEnzyme MLIRLLVMExt --parallel 4
```
