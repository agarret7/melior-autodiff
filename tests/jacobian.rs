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
use common::{activity_array_attr, gen_jacobian, setup_context};

fn parse_square_jacobian_module(ctx: &melior::Context) -> Module<'_> {
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
    .expect("failed to parse jacobian test module");
    gen_jacobian(
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

#[test]
fn jacobian_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let dx = entry.argument(1).unwrap().into();
    let result_types = [f64_ty, f64_ty];
    let inputs = [x, dx];
    let jac_op = OperationBuilder::new("enzyme.jacobian", loc)
        .add_operands(&inputs)
        .add_results(&result_types)
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

    let ret = OperationBuilder::new("func.return", loc)
        .add_operands(&[primal])
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
                StringAttribute::new(&ctx, "fwd_caller").into(),
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
    eprintln!("jacobian_op_with_inputs:\n{text}");
    assert!(text.contains("enzyme.jacobian"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dupnoneed"));
    assert!(text.contains("enzyme_active"));
    assert!(text.contains("enzyme_constnoneed"));
    assert!(text.contains("strong_zero = true"));
    assert!(text.contains("f64, f64") || text.contains("%arg0") && text.contains("%arg1"));
}

// enzyme.jacobian exists in the Enzyme dialect and can be constructed, but no
// Enzyme pass currently lowers it (EnzymeMLIRPass only handles autodiff/fwddiff).
// Re-enable once upstream adds a jacobian lowering pass.
#[test]
#[ignore]
fn differentiate_pass_lowers_jacobian() {
    let ctx = setup_context();
    let mut module = parse_square_jacobian_module(&ctx);

    eprintln!("before differentiation:\n{}", module.as_operation());

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });

    let result = pm.run(&mut module);
    let text = module.as_operation().to_string();
    eprintln!("after jacobian differentiation:\n{text}");
    result.expect("Enzyme jacobian pass failed");

    assert!(
        !text.contains("enzyme.jacobian"),
        "enzyme.jacobian was not lowered"
    );
    assert!(
        text.contains("arith.mulf"),
        "lowered jacobian should contain floating-point multiply"
    );
    assert!(
        !text.contains("arith.muli"),
        "floating multiply was corrupted into integer multiply"
    );
}
