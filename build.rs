use std::path::PathBuf;

fn main() {
    #[cfg(feature = "enzyme")]
    build_enzyme_bindings();
}

#[cfg(feature = "enzyme")]
fn build_enzyme_bindings() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let enzyme_src = manifest.join("enzyme/enzyme/Enzyme/MLIR/Integrations/c");

    // Allow overriding the build dir via env var; default to our submodule build.
    let enzyme_build = std::env::var("ENZYME_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("enzyme/build"));

    let lib_dir = enzyme_build.join("Enzyme/MLIR/Integrations/c");

    let llvm_prefix = std::env::var("MLIR_SYS_230_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/usr/lib/llvm-23"));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // MLIRCAPIEnzyme depends on these MLIR/Enzyme libs.
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Dialect").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Dialect/Impulse").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Interfaces").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Analysis").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Passes").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Implementations").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        enzyme_build.join("Enzyme/MLIR/Dialect/LLVMExt").display()
    );
    println!("cargo:rustc-link-lib=static=MLIRCAPIEnzyme");
    println!("cargo:rustc-link-lib=static=MLIREnzyme");
    println!("cargo:rustc-link-lib=static=MLIRImpulse");
    println!("cargo:rustc-link-lib=static=MLIREnzymeAutoDiffInterface");
    println!("cargo:rustc-link-lib=static=MLIREnzymeAnalysis");
    println!("cargo:rustc-link-lib=static=MLIREnzymeTransforms");
    println!("cargo:rustc-link-lib=static=MLIREnzymeImplementations");
    println!("cargo:rustc-link-lib=static=MLIRLLVMExt");
    println!(
        "cargo:rustc-link-search=native={}",
        llvm_prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=MLIRCAPIIR");
    println!("cargo:rustc-link-lib=stdc++");

    println!(
        "cargo:rerun-if-changed={}",
        enzyme_src.join("EnzymeMLIR.h").display()
    );
    // Relink when a rebuilt Enzyme library changes without a header change.
    for lib in [
        "Integrations/c/libMLIRCAPIEnzyme.a",
        "Dialect/libMLIREnzyme.a",
        "Dialect/Impulse/libMLIRImpulse.a",
        "Interfaces/libMLIREnzymeAutoDiffInterface.a",
        "Analysis/libMLIREnzymeAnalysis.a",
        "Passes/libMLIREnzymeTransforms.a",
        "Implementations/libMLIREnzymeImplementations.a",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            enzyme_build.join("Enzyme/MLIR").join(lib).display()
        );
    }

    let bindings = bindgen::Builder::default()
        .header(enzyme_src.join("EnzymeMLIR.h").to_str().unwrap())
        .clang_arg(format!("-I{}", llvm_prefix.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            enzyme_build.join("Enzyme/MLIR/Dialect").display()
        ))
        // Reuse mlir_sys types rather than re-generating them.
        .blocklist_type("Mlir.*")
        .raw_line("use mlir_sys::*;")
        .allowlist_function("enzymeActivity.*|enzymeRegisterDialectExtensions|enzymeCreate.*|enzymeConvert.*|enzyme.*OpCreate|mlirGetDialectHandle__enzyme__|mlirGetDialectHandle__impulse__")
        .allowlist_type("")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on EnzymeMLIR.h");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("enzyme_bindings.rs"))
        .unwrap();
}
