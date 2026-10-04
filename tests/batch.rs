use melior::ir::{Location, Module};
use melior::pass::PassManager;
use melior_autodiff::{
    create_context, enzymeCreateBatchDiffPass, enzymeCreateBatchPass,
    enzymeCreateRemoveUnusedEnzymeOpsPass,
};

// Runs enzyme-batch on a parsed module and checks that the batch op is lowered
// to a call to a generated @batched_* function.
#[test]
fn batch_pass_lowers_batch_op() {
    let ctx = create_context();
    let mut module = Module::parse(
        &ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %y = math.sin %x : f64
    return %y : f64
  }
  func.func @dsquare(%x: tensor<4xf64>) -> tensor<4xf64> {
    %r = enzyme.batch @square(%x) { batch_shape = array<i64: 4> }
      : (tensor<4xf64>) -> tensor<4xf64>
    return %r : tensor<4xf64>
  }
}
"#,
    )
    .expect("failed to parse batch test module");

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateBatchPass) });
    pm.run(&mut module).expect("enzyme-batch pass failed");

    let text = module.as_operation().to_string();
    assert!(
        !text.contains("enzyme.batch"),
        "enzyme.batch should be lowered; got:\n{text}"
    );
    assert!(
        text.contains("batched_square"),
        "expected @batched_square to be generated; got:\n{text}"
    );
}

// Verifies that all three new pass constructors produce passes that run
// without panicking, even on an empty module.
#[test]
fn batch_and_cleanup_passes_are_runnable() {
    let ctx = create_context();
    let mut module = Module::new(Location::unknown(&ctx));
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateBatchPass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateBatchDiffPass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateRemoveUnusedEnzymeOpsPass) });
    pm.run(&mut module)
        .expect("batch and cleanup passes should run on an empty module");
}
