use melior::ir::{
    attribute::{
        BoolAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    r#type::IntegerType,
    Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type,
};
use melior::pass::PassManager;
use melior_autodiff::{enzymeCreateDifferentiatePass, Activity};
use mlir_sys::{mlirF64TypeGet, mlirFunctionTypeGet};

mod common;
use common::{activity_array_attr, gen_autodiff, setup_context};

fn parse_square_autodiff_module(ctx: &melior::Context) -> Module<'_> {
    let module = Module::parse(
        ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }
}
"#,
    )
    .expect("failed to parse test module");
    gen_autodiff(
        ctx,
        &module,
        "dsquare",
        "square",
        &["f64", "f64"],
        &["f64"],
        &[Activity::Active],
        &[Activity::ActiveNoNeed],
        false,
    );
    module
}

fn parse_tensor_autodiff_module(ctx: &melior::Context) -> Module<'_> {
    let module = Module::parse(
        ctx,
        r#"
module {
  func.func @square_tensor(%x: tensor<2xf64>) -> tensor<2xf64> {
    %y = arith.mulf %x, %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }
}
"#,
    )
    .expect("failed to parse tensor autodiff test module");
    gen_autodiff(
        ctx,
        &module,
        "dsquare_tensor",
        "square_tensor",
        &["tensor<2xf64>", "tensor<2xf64>"],
        &["tensor<2xf64>"],
        &[Activity::Active],
        &[Activity::ActiveNoNeed],
        false,
    );
    module
}

/// Minimal test: no inputs, just verifies the op constructs and serializes.
#[test]
fn autodiff_op_no_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let result_types = [Type::float64(&ctx)];
    let autodiff_op = OperationBuilder::new("enzyme.autodiff", loc)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "fn"),
                FlatSymbolRefAttribute::new(&ctx, "square").into(),
            ),
            (
                Identifier::new(&ctx, "activity"),
                activity_array_attr(&ctx, &[Activity::Dup]).into(),
            ),
            (
                Identifier::new(&ctx, "ret_activity"),
                activity_array_attr(&ctx, &[Activity::Active]).into(),
            ),
            (
                Identifier::new(&ctx, "width"),
                IntegerAttribute::new(IntegerType::new(&ctx, 64).into(), 1).into(),
            ),
            (
                Identifier::new(&ctx, "strong_zero"),
                BoolAttribute::new(&ctx, false).into(),
            ),
        ])
        .build()
        .unwrap();

    module.body().append_operation(autodiff_op);

    let text = module.as_operation().to_string();
    eprintln!("autodiff_op_no_inputs:\n{text}");
    assert!(text.contains("enzyme.autodiff"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dup"));
}

/// More realistic test: autodiff op with primal + shadow inputs, embedded
/// inside a func.func so SSA values are properly numbered.
#[test]
fn autodiff_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let f64_ty = Type::float64(&ctx);

    // Build a caller function: (%x: f64, %dx: f64) -> f64
    // The body calls enzyme.autodiff @square(%x, %dx) and returns the result.
    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let dx = entry.argument(1).unwrap().into();
    let result_types = [f64_ty];
    let inputs = [x, dx];
    let autodiff_op = OperationBuilder::new("enzyme.autodiff", loc)
        .add_operands(&inputs)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "fn"),
                FlatSymbolRefAttribute::new(&ctx, "square").into(),
            ),
            (
                Identifier::new(&ctx, "activity"),
                activity_array_attr(&ctx, &[Activity::Dup]).into(),
            ),
            (
                Identifier::new(&ctx, "ret_activity"),
                activity_array_attr(&ctx, &[Activity::Active]).into(),
            ),
            (
                Identifier::new(&ctx, "width"),
                IntegerAttribute::new(IntegerType::new(&ctx, 64).into(), 1).into(),
            ),
            (
                Identifier::new(&ctx, "strong_zero"),
                BoolAttribute::new(&ctx, false).into(),
            ),
        ])
        .build()
        .unwrap();

    let result = entry
        .append_operation(autodiff_op)
        .result(0)
        .unwrap()
        .into();

    let ret = OperationBuilder::new("func.return", loc)
        .add_operands(&[result])
        .build()
        .unwrap();
    entry.append_operation(ret);

    let body = Region::new();
    body.append_block(entry);

    let fn_type = unsafe {
        let raw_ctx = ctx.to_raw();
        let args = [mlirF64TypeGet(raw_ctx), mlirF64TypeGet(raw_ctx)];
        let rets = [mlirF64TypeGet(raw_ctx)];
        Type::from_raw(mlirFunctionTypeGet(
            raw_ctx,
            2,
            args.as_ptr(),
            1,
            rets.as_ptr(),
        ))
    };

    let caller_fn = OperationBuilder::new("func.func", loc)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "sym_name"),
                StringAttribute::new(&ctx, "caller").into(),
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
    eprintln!("autodiff_op_with_inputs:\n{text}");
    assert!(text.contains("enzyme.autodiff"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dup"));
    // Both inputs should appear as operands in the serialized output.
    assert!(text.contains("f64, f64") || text.contains("%arg0") && text.contains("%arg1"));
}

/// Run the Enzyme differentiate pass on a module containing @square and an
/// enzyme.jacobian op matching Enzyme's square.mlir test, and verify the
/// pass lowers it to concrete IR.
///
/// Mirrors:
///   func.func @dsquare(%x: f64, %dr: f64) -> f64 {
///     %r = enzyme.jacobian @square(%x, %dr) {
///       activity=[enzyme_active], ret_activity=[enzyme_activenoneed]
///     } : (f64, f64) -> f64
///     return %r : f64
///   }
#[test]
fn differentiate_pass_lowers_autodiff() {
    let ctx = setup_context();
    let mut module = parse_square_autodiff_module(&ctx);

    eprintln!("before differentiation:\n{}", module.as_operation());

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    let result = pm.run(&mut module);
    let text = module.as_operation().to_string();
    eprintln!("after differentiation:\n{text}");
    result.expect("Enzyme differentiate pass failed");

    assert!(
        !text.contains("enzyme.autodiff"),
        "enzyme.autodiff was not lowered"
    );
    assert!(
        text.contains("call @diffesquare(%arg0, %arg1) : (f64, f64) -> f64"),
        "@dsquare does not call generated @diffesquare"
    );
    assert!(
        text.contains("func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64"),
        "generated @diffesquare function is missing"
    );
    assert!(
        text.contains(
            "func.func @square(%arg0: f64) -> f64 {\n    %0 = arith.mulf %arg0, %arg0 : f64"
        ),
        "original @square was not preserved as arith.mulf"
    );
    assert!(
        !text.contains("arith.muli"),
        "floating multiply was corrupted into integer multiply"
    );
}

/// Concrete ranked tensor smoke test for Enzyme's reverse-mode tensor AD
/// interface. This catches the tensor gradient/cache path, not only
/// forward-mode tangent propagation.
#[test]
fn differentiate_pass_lowers_tensor_autodiff() {
    let ctx = setup_context();
    let mut module = parse_tensor_autodiff_module(&ctx);

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });

    let result = pm.run(&mut module);
    let text = module.as_operation().to_string();
    eprintln!("after tensor autodiff differentiation:\n{text}");
    result.expect("Enzyme tensor autodiff pass failed");

    assert!(
        !text.contains("enzyme.autodiff"),
        "enzyme.autodiff was not lowered"
    );
    assert!(
        text.contains("call @diffesquare_tensor(%arg0, %arg1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>"),
        "@dsquare_tensor does not call generated tensor derivative"
    );
    assert!(
        text.contains("func.func private @diffesquare_tensor(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64>"),
        "generated tensor reverse derivative function is missing"
    );
    assert!(
        text.contains("!enzyme.Gradient<tensor<2xf64>>"),
        "tensor reverse derivative should use tensor gradient storage"
    );
    assert!(
        text.contains("arith.mulf") && text.contains("tensor<2xf64>"),
        "tensor reverse derivative should contain tensor multiplies"
    );
    assert!(
        !text.contains("arith.muli"),
        "floating tensor multiply was corrupted into integer multiply"
    );
}
