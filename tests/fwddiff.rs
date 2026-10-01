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
use common::{activity_array_attr, gen_fwddiff, setup_context};

fn parse_tensor_fwddiff_module(ctx: &melior::Context) -> Module<'_> {
    let module = Module::parse(
        ctx,
        r#"
module {
  func.func @sin_tensor(%x: tensor<2xf64>) -> tensor<2xf64> {
    %y = math.sin %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }
}
"#,
    )
    .expect("failed to parse tensor fwddiff test module");
    gen_fwddiff(
        ctx,
        &module,
        "dsin_tensor",
        "sin_tensor",
        &["tensor<2xf64>", "tensor<2xf64>"],
        &["tensor<2xf64>"],
        &[Activity::Dup],
        &[Activity::DupNoNeed],
        false,
    );
    module
}

/// Forward-mode sibling of enzyme.autodiff: verify the C API constructs an
/// enzyme.fwddiff op with operands and activity metadata intact.
#[test]
fn forwarddiff_op_with_inputs() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let f64_ty = Type::float64(&ctx);

    let entry = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let x = entry.argument(0).unwrap().into();
    let dx = entry.argument(1).unwrap().into();
    let result_types = [f64_ty, f64_ty];
    let inputs = [x, dx];
    let fwddiff_op = OperationBuilder::new("enzyme.fwddiff", loc)
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

    let primal = entry.append_operation(fwddiff_op).result(0).unwrap().into();

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
    eprintln!("forwarddiff_op_with_inputs:\n{text}");
    assert!(text.contains("enzyme.fwddiff"));
    assert!(text.contains("@square"));
    assert!(text.contains("enzyme_dupnoneed"));
    assert!(text.contains("enzyme_active"));
    assert!(text.contains("enzyme_constnoneed"));
    assert!(text.contains("strong_zero = true"));
    assert!(text.contains("f64, f64") || text.contains("%arg0") && text.contains("%arg1"));
}

/// Concrete ranked tensor smoke test for Enzyme's forward-mode tensor AD
/// interface. This mirrors Enzyme's tensor<2xf64> math.sin coverage and
/// verifies the pass generates an outlined tensor derivative.
#[test]
fn differentiate_pass_lowers_tensor_fwddiff() {
    let ctx = setup_context();
    let mut module = parse_tensor_fwddiff_module(&ctx);

    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });

    let result = pm.run(&mut module);
    let text = module.as_operation().to_string();
    eprintln!("after tensor fwddiff differentiation:\n{text}");
    result.expect("Enzyme tensor fwddiff pass failed");

    assert!(
        !text.contains("enzyme.fwddiff"),
        "enzyme.fwddiff was not lowered"
    );
    assert!(
        text.contains("call @fwddiffesin_tensor(%arg0, %arg1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>"),
        "@dsin_tensor does not call generated tensor derivative"
    );
    assert!(
        text.contains("func.func private @fwddiffesin_tensor(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64>"),
        "generated tensor derivative function is missing"
    );
    assert!(
        text.contains("math.cos %arg0 : tensor<2xf64>"),
        "tensor derivative should contain cos(primal)"
    );
    assert!(
        text.contains("arith.mulf") && text.contains("tensor<2xf64>"),
        "tensor derivative should multiply tangent by cos(primal)"
    );
}
