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

fn parse_square_module(ctx: &melior::Context) -> Module<'_> {
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

fn parse_square_tensor_module(ctx: &melior::Context) -> Module<'_> {
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
    .expect("failed to parse tensor test module");
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

// ── Op construction ───────────────────────────────────────────────────────────

#[test]
fn autodiff_op_no_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let autodiff_op = OperationBuilder::new("enzyme.autodiff", loc)
        .add_results(&[Type::float64(&ctx)])
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
    assert!(text.contains("enzyme.autodiff"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dup"));
}

#[test]
fn autodiff_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let dx = entry.argument(1).unwrap().into();
    let autodiff_op = OperationBuilder::new("enzyme.autodiff", loc)
        .add_operands(&[x, dx])
        .add_results(&[f64_ty])
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

    let result = entry.append_operation(autodiff_op).result(0).unwrap().into();
    entry.append_operation(
        OperationBuilder::new("func.return", loc)
            .add_operands(&[result])
            .build()
            .unwrap(),
    );

    let body = Region::new();
    body.append_block(entry);

    let fn_type = unsafe {
        let raw_ctx = ctx.to_raw();
        let args = [mlirF64TypeGet(raw_ctx), mlirF64TypeGet(raw_ctx)];
        let rets = [mlirF64TypeGet(raw_ctx)];
        Type::from_raw(mlirFunctionTypeGet(raw_ctx, 2, args.as_ptr(), 1, rets.as_ptr()))
    };

    module.body().append_operation(
        OperationBuilder::new("func.func", loc)
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
            .unwrap(),
    );

    let text = module.as_operation().to_string();
    assert!(text.contains("enzyme.autodiff"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dup"));
}

#[test]
fn jacobian_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let dx = entry.argument(1).unwrap().into();
    let jac_op = OperationBuilder::new("enzyme.jacobian", loc)
        .add_operands(&[x, dx])
        .add_results(&[f64_ty, f64_ty])
        .add_attributes(&[
            (
                Identifier::new(&ctx, "fn"),
                FlatSymbolRefAttribute::new(&ctx, "square").into(),
            ),
            (
                Identifier::new(&ctx, "activity"),
                activity_array_attr(&ctx, &[Activity::DupNoNeed]).into(),
            ),
            (
                Identifier::new(&ctx, "ret_activity"),
                activity_array_attr(&ctx, &[Activity::Active, Activity::ConstNoNeed]).into(),
            ),
            (
                Identifier::new(&ctx, "width"),
                IntegerAttribute::new(IntegerType::new(&ctx, 64).into(), 1).into(),
            ),
            (
                Identifier::new(&ctx, "strong_zero"),
                BoolAttribute::new(&ctx, true).into(),
            ),
        ])
        .build()
        .unwrap();

    let primal = entry.append_operation(jac_op).result(0).unwrap().into();
    entry.append_operation(
        OperationBuilder::new("func.return", loc)
            .add_operands(&[primal])
            .build()
            .unwrap(),
    );

    let body = Region::new();
    body.append_block(entry);

    let fn_type = unsafe {
        let raw_ctx = ctx.to_raw();
        let args = [mlirF64TypeGet(raw_ctx), mlirF64TypeGet(raw_ctx)];
        let rets = [mlirF64TypeGet(raw_ctx)];
        Type::from_raw(mlirFunctionTypeGet(raw_ctx, 2, args.as_ptr(), 1, rets.as_ptr()))
    };

    module.body().append_operation(
        OperationBuilder::new("func.func", loc)
            .add_attributes(&[
                (
                    Identifier::new(&ctx, "sym_name"),
                    StringAttribute::new(&ctx, "jac_caller").into(),
                ),
                (
                    Identifier::new(&ctx, "function_type"),
                    TypeAttribute::new(fn_type).into(),
                ),
            ])
            .add_regions([body])
            .build()
            .unwrap(),
    );

    let text = module.as_operation().to_string();
    assert!(text.contains("enzyme.jacobian"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dupnoneed"));
    assert!(text.contains("enzyme_active"));
    assert!(text.contains("enzyme_constnoneed"));
    assert!(text.contains("strong_zero = true"));
}

// ── Differentiate pass ────────────────────────────────────────────────────────

#[test]
fn differentiate_pass_lowers_autodiff() {
    let ctx = setup_context();
    let mut module = parse_square_module(&ctx);

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(&mut module).expect("Enzyme differentiate pass failed");

    let text = module.as_operation().to_string();
    assert!(!text.contains("enzyme.autodiff"), "enzyme.autodiff was not lowered");
    assert!(text.contains("call @diffesquare(%arg0, %arg1) : (f64, f64) -> f64"));
    assert!(text.contains("func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64"));
    assert!(!text.contains("arith.muli"));
}

#[test]
fn differentiate_pass_lowers_tensor_autodiff() {
    let ctx = setup_context();
    let mut module = parse_square_tensor_module(&ctx);

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(&mut module).expect("Enzyme tensor autodiff pass failed");

    let text = module.as_operation().to_string();
    assert!(!text.contains("enzyme.autodiff"), "enzyme.autodiff was not lowered");
    assert!(text.contains("!enzyme.Gradient<tensor<2xf64>>"));
    assert!(text.contains("arith.mulf") && text.contains("tensor<2xf64>"));
    assert!(!text.contains("arith.muli"));
}

// enzyme.jacobian op exists in the dialect but no pass currently lowers it.
#[test]
#[ignore]
fn differentiate_pass_lowers_jacobian() {
    let ctx = setup_context();
    let mut module = parse_square_module(&ctx);

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(&mut module).expect("Enzyme jacobian pass failed");

    let text = module.as_operation().to_string();
    assert!(!text.contains("enzyme.jacobian"), "enzyme.jacobian was not lowered");
}
