use melior::pass::PassManager;
use melior::{
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type, ValueLike,
    },
};
use melior_autodiff::{
    activity_attr, create_jacobian_op, enzymeCreateDifferentiatePass,
    enzymeRegisterPasses, Activity,
};
use mlir_sys::{
    mlirF64TypeGet, mlirFunctionTypeGet, mlirLocationUnknownGet,
};

mod common;
use common::setup_context;
use common::parse_square_jacobian_module;

#[test]
fn jacobian_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().to_raw();
    let dx = entry.argument(1).unwrap().to_raw();

    let jac_op = unsafe {
        let raw_ctx = ctx.to_raw();
        let raw_loc = mlirLocationUnknownGet(raw_ctx);
        let result_types = [mlirF64TypeGet(raw_ctx), mlirF64TypeGet(raw_ctx)];
        let inputs = [x, dx];
        let activity_arr = [activity_attr(raw_ctx, Activity::DupNoNeed)];
        let ret_activity_arr = [
            activity_attr(raw_ctx, Activity::Active),
            activity_attr(raw_ctx, Activity::ConstNoNeed),
        ];

        create_jacobian_op(
            raw_ctx,
            "square",
            &result_types,
            &inputs,
            &activity_arr,
            &ret_activity_arr,
            1,
            true,
            raw_loc,
        )
    };
    assert!(!jac_op.ptr.is_null());

    let primal = entry
        .append_operation(unsafe { melior::ir::Operation::from_raw(jac_op) })
        .result(0)
        .unwrap()
        .into();

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

    unsafe { enzymeRegisterPasses() };
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