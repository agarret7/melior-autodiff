use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// The Enzyme libraries this crate links, by their directory under `Enzyme/MLIR` in a build.
const ENZYME_LIBS: &[(&str, &str)] = &[
    ("Integrations/c", "MLIRCAPIEnzyme"),
    ("Dialect", "MLIREnzyme"),
    ("Dialect/Impulse", "MLIRImpulse"),
    ("Dialect/LLVMExt", "MLIRLLVMExt"),
    ("Interfaces", "MLIREnzymeAutoDiffInterface"),
    ("Analysis", "MLIREnzymeAnalysis"),
    ("Passes", "MLIREnzymeTransforms"),
    ("Implementations", "MLIREnzymeImplementations"),
];

/// The CMake targets that build every library in `ENZYME_LIBS`.
const ENZYME_TARGETS: &[&str] = &["MLIRCAPIEnzyme", "MLIRLLVMExt"];

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let enzyme_src = manifest.join("enzyme/enzyme");
    let llvm_prefix = llvm_prefix();
    // The FileCheck tests run the FileCheck from this LLVM.
    println!("cargo:rustc-env=LLVM_PREFIX={}", llvm_prefix.display());

    // `ods::enzyme` generates the dialect's ops from its TableGen definitions here.
    let enzyme_dialect = enzyme_src.join("Enzyme/MLIR/Dialect");
    println!(
        "cargo:rustc-env=ENZYME_DIALECT_DIR={}",
        enzyme_dialect.display()
    );
    for td in ["EnzymeOps.td", "Dialect.td", "EnzymeEnums.td"] {
        println!(
            "cargo:rerun-if-changed={}",
            enzyme_dialect.join(td).display()
        );
    }

    // An existing Enzyme build, or else one of the submodule, built here.
    println!("cargo:rerun-if-env-changed=ENZYME_BUILD_DIR");
    let enzyme_build = match env::var_os("ENZYME_BUILD_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => build_enzyme(&enzyme_src, &llvm_prefix),
    };

    link(&enzyme_build, &llvm_prefix);
    generate_bindings(&enzyme_src, &enzyme_build, &llvm_prefix);
    list_pass_constructors(&enzyme_build);
}

/// Passes the tests every pass constructor Enzyme generates, as `ENZYME_PASS_CONSTRUCTORS`, so they
/// can check that each one is wrapped.
fn list_pass_constructors(enzyme_build: &Path) {
    let header = enzyme_build.join("Enzyme/MLIR/Passes/Passes.capi.h.inc");
    let header = std::fs::read_to_string(&header)
        .unwrap_or_else(|error| panic!("couldn't read {}: {error}", header.display()));
    let constructors: Vec<&str> = header
        .split_whitespace()
        .filter_map(|word| word.strip_suffix("(void);"))
        .filter(|name| name.starts_with("mlirCreateEnzyme"))
        .collect();
    println!(
        "cargo:rustc-env=ENZYME_PASS_CONSTRUCTORS={}",
        constructors.join(",")
    );
}

/// The LLVM install to build against: `MLIR_SYS_230_PREFIX`, or else the one whose `llvm-config`
/// is on `PATH`, the same one `mlir-sys` links.
fn llvm_prefix() -> PathBuf {
    println!("cargo:rerun-if-env-changed=MLIR_SYS_230_PREFIX");
    if let Some(prefix) = env::var_os("MLIR_SYS_230_PREFIX") {
        return PathBuf::from(prefix);
    }
    let output = Command::new("llvm-config")
        .arg("--prefix")
        .output()
        .unwrap_or_else(|error| {
            panic!("set MLIR_SYS_230_PREFIX to your LLVM 23 install, or put its llvm-config on PATH: {error}")
        });
    if !output.status.success() {
        panic!("llvm-config --prefix exited with {}", output.status);
    }
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

/// Builds Enzyme's MLIR C API from `src` into `OUT_DIR`, against the LLVM at `llvm_prefix`, and
/// returns the build directory.
fn build_enzyme(src: &Path, llvm_prefix: &Path) -> PathBuf {
    if !src.join("CMakeLists.txt").exists() {
        panic!(
            "Enzyme's sources are missing from {}: run `git submodule update --init`, or set \
             ENZYME_BUILD_DIR to an existing Enzyme build",
            src.display()
        );
    }
    for path in ["CMakeLists.txt", "cmake", "include", "Enzyme", "tools"] {
        println!("cargo:rerun-if-changed={}", src.join(path).display());
    }
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");

    let build = PathBuf::from(env::var("OUT_DIR").unwrap()).join("enzyme");
    if !build.join("CMakeCache.txt").exists() {
        let mut configure = Command::new("cmake");
        configure
            .arg("-S")
            .arg(src)
            .arg("-B")
            .arg(&build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!(
                "-DLLVM_DIR={}",
                llvm_prefix.join("lib/cmake/llvm").display()
            ))
            .arg(format!(
                "-DMLIR_DIR={}",
                llvm_prefix.join("lib/cmake/mlir").display()
            ))
            .arg("-DENZYME_MLIR=ON")
            .arg("-DENZYME_CLANG=OFF")
            .arg("-DENZYME_BC_LOADER=OFF");
        // Enzyme has to be compiled by a compiler that can build against this LLVM, so default to
        // the clang installed with it.
        for (var, name, clang) in [("CC", "C", "clang"), ("CXX", "CXX", "clang++")] {
            let clang = llvm_prefix.join("bin").join(clang);
            if env::var_os(var).is_none() && clang.exists() {
                configure.arg(format!("-DCMAKE_{name}_COMPILER={}", clang.display()));
            }
        }
        run(&mut configure, "configure Enzyme");
    }

    // Each job can take a couple of gigabytes, so use at most 4 unless ENZYME_BUILD_JOBS says
    // otherwise, and fewer if Cargo runs fewer.
    println!("cargo:rerun-if-env-changed=ENZYME_BUILD_JOBS");
    let jobs = env::var("ENZYME_BUILD_JOBS").unwrap_or_else(|_| {
        let cargo_jobs: usize = env::var("NUM_JOBS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(1);
        cargo_jobs.min(4).to_string()
    });
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build)
            .arg("--target")
            .args(ENZYME_TARGETS)
            .args(["--parallel", &jobs]),
        "build Enzyme",
    );
    build
}

fn run(command: &mut Command, what: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {what}: couldn't run cmake: {error}"));
    if !status.success() {
        panic!("failed to {what}: {command:?} exited with {status}");
    }
}

fn link(enzyme_build: &Path, llvm_prefix: &Path) {
    let enzyme_mlir = enzyme_build.join("Enzyme/MLIR");
    for (dir, lib) in ENZYME_LIBS {
        println!(
            "cargo:rustc-link-search=native={}",
            enzyme_mlir.join(dir).display()
        );
        println!("cargo:rustc-link-lib=static={lib}");
        // Relink when a rebuilt library changes without a header change.
        println!(
            "cargo:rerun-if-changed={}",
            enzyme_mlir.join(dir).join(format!("lib{lib}.a")).display()
        );
    }
    println!(
        "cargo:rustc-link-search=native={}",
        llvm_prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=MLIRCAPIIR");
}

fn generate_bindings(enzyme_src: &Path, enzyme_build: &Path, llvm_prefix: &Path) {
    let capi = enzyme_src.join("Enzyme/MLIR/Integrations/c");
    for header in [
        capi.join("EnzymeMLIR.h"),
        capi.join("EnzymeMLIRPasses.h"),
        enzyme_build.join("Enzyme/MLIR/Passes/Passes.capi.h.inc"),
    ] {
        println!("cargo:rerun-if-changed={}", header.display());
    }

    let bindings = bindgen::Builder::default()
        .header(capi.join("EnzymeMLIR.h").to_str().unwrap())
        // The pass constructors Enzyme generates from Passes.td, declared in an included `.inc`
        // under the build directory.
        .header(capi.join("EnzymeMLIRPasses.h").to_str().unwrap())
        .clang_arg(format!("-I{}", llvm_prefix.join("include").display()))
        .clang_arg(format!("-I{}", enzyme_build.join("Enzyme/MLIR").display()))
        .clang_arg(format!(
            "-I{}",
            enzyme_build.join("Enzyme/MLIR/Dialect").display()
        ))
        // Reuse mlir_sys types rather than re-generating them.
        .blocklist_type("Mlir.*")
        .raw_line("use mlir_sys::*;")
        .allowlist_function("enzymeActivity.*|enzymeRegisterDialectExtensions|enzymeCreate.*|mlirCreateEnzyme.*|mlirRegisterEnzymePasses|mlirGetDialectHandle__enzyme__")
        .allowlist_type("")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on EnzymeMLIR.h");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("enzyme_bindings.rs"))
        .unwrap();
}
