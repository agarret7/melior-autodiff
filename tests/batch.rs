use melior::ir::{
    attribute::{DenseI64ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
    operation::OperationBuilder,
    Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type,
};
use melior::pass::PassManager;
use melior_autodiff::{
    enzymeCreateBatchDiffPass, enzymeCreateBatchPass, enzymeCreateRemoveUnusedEnzymeOpsPass,
};
use mlir_sys::{mlirF64TypeGet, mlirFunctionTypeGet};

mod common;
use common::{gen_batch, setup_context};

#[test]
fn batch_op_construction() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let result_types = [f64_ty];
    let inputs = [x];
    let batch_op = OperationBuilder::new("enzyme.batch", loc)
        .add_operands(&inputs)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "fn"),
                FlatSymbolRefAttribute::new(&ctx, "square").into(),
            ),
            (
                Identifier::new(&ctx, "batch_shape"),
                DenseI64ArrayAttribute::new(&ctx, &[4]).into(),
            ),
        ])
        .build()
        .unwrap();

    let result = entry.append_operation(batch_op).result(0).unwrap().into();

    let ret = OperationBuilder::new("func.return", loc)
        .add_operands(&[result])
        .build()
        .unwrap();
    entry.append_operation(ret);

    let body = Region::new();
    body.append_block(entry);

    let fn_type = unsafe {
        let raw_ctx = ctx.to_raw();
        let args = [mlirF64TypeGet(raw_ctx)];
        let rets = [mlirF64TypeGet(raw_ctx)];
        Type::from_raw(mlirFunctionTypeGet(
            raw_ctx,
            1,
            args.as_ptr(),
            1,
            rets.as_ptr(),
        ))
    };

    let caller_fn = OperationBuilder::new("func.func", loc)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "sym_name"),
                StringAttribute::new(&ctx, "batch_caller").into(),
            ),
            (
                Identifier::new(&ctx, "function_type"),
                TypeAttribute::new(fn_type).into(),
            ),
        ])
        .add_regions([body])
        .build()
        .unwrap();
    module.body().append_operation(caller_fn);

    let text = module.as_operation().to_string();
    assert!(text.contains("enzyme.batch"));
    assert!(text.contains("@square"));
    assert!(text.contains("batch_shape = array<i64: 4>"));
}

// Runs enzyme-batch on a parsed module and checks that the batch op is lowered
// to a call to a generated @batched_* function.
#[test]
fn batch_pass_lowers_batch_op() {
    let ctx = setup_context();

    let mut module = Module::parse(
        &ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %y = math.sin %x : f64
    return %y : f64
  }
}
"#,
    )
    .expect("failed to parse batch test module");
    gen_batch(
        &ctx,
        &module,
        "dsquare",
        "square",
        &["tensor<4xf64>"],
        &["tensor<4xf64>"],
        &[4],
    );

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
    let ctx = setup_context();

    let mut module = Module::new(Location::unknown(&ctx));
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateBatchPass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateBatchDiffPass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateRemoveUnusedEnzymeOpsPass) });
    pm.run(&mut module)
        .expect("batch and cleanup passes should run on an empty module");
}
