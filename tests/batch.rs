use melior::ir::{Location, Module};
use melior::pass::PassManager;
use melior_autodiff::{create_context, pass};

mod filecheck;
use filecheck::filecheck;

// Runs enzyme-batch on a parsed module and checks that the batch op is lowered
// to a call to a generated @batched_* function.
#[test]
fn batch_pass_lowers_batch_op() {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, include_str!("mlir/batch/square.mlir"))
        .expect("failed to parse batch test module");

    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.run(&mut module).expect("enzyme-batch pass failed");

    filecheck(&module.as_operation().to_string(), "mlir/batch/square.mlir", None);
}

// Verifies that all three new pass constructors produce passes that run
// without panicking, even on an empty module.
#[test]
fn batch_and_cleanup_passes_are_runnable() {
    let ctx = create_context();
    let mut module = Module::new(Location::unknown(&ctx));
    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.add_pass(pass::create_batch_diff_pass());
    pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
    pm.run(&mut module)
        .expect("batch and cleanup passes should run on an empty module");
}
