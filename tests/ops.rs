use melior::dialect::func;
use melior::ir::{
    attribute::{StringAttribute, TypeAttribute},
    operation::OperationLike,
    r#type::FunctionType,
    Block, BlockLike, Location, Module, Operation, Region, RegionLike, Type, Value,
};
use melior::pass::PassManager;
use melior::Context;
use melior_autodiff::{
    activity_array_attribute, autodiff, batch, create_context, enzymeCreateDifferentiatePass,
    fwddiff, jacobian, Activity,
};

mod filecheck;
use filecheck::filecheck;

const SQUARE: &str = "mlir/ops/square.mlir";
const SQUARE_TENSOR: &str = "mlir/ops/square_tensor.mlir";

// Parses tests/<file> and appends `func.func @caller` whose body is the single op built by `build`.
fn with_caller<'c>(
    ctx: &'c Context,
    file: &str,
    arg_types: &[Type<'c>],
    result_types: &[Type<'c>],
    build: impl FnOnce(&[Value<'c, '_>]) -> Operation<'c>,
) -> Module<'c> {
    let src = std::fs::read_to_string(format!("{}/tests/{file}", env!("CARGO_MANIFEST_DIR")))
        .expect("missing test file");
    let module = Module::parse(ctx, &src).expect("parse failed");
    let loc = Location::unknown(ctx);
    let block = Block::new(&arg_types.iter().map(|&t| (t, loc)).collect::<Vec<_>>());
    {
        let args = (0..arg_types.len())
            .map(|i| block.argument(i).unwrap().into())
            .collect::<Vec<_>>();
        let op = block.append_operation(build(&args));
        let results = (0..result_types.len())
            .map(|i| op.result(i).unwrap().into())
            .collect::<Vec<_>>();
        block.append_operation(func::r#return(&results, loc));
    }
    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        ctx,
        StringAttribute::new(ctx, "caller"),
        TypeAttribute::new(FunctionType::new(ctx, arg_types, result_types).into()),
        region,
        &[],
        loc,
    ));
    assert!(module.as_operation().verify(), "module failed to verify");
    module
}

fn differentiate(ctx: &Context, module: &mut Module) -> String {
    let pm = PassManager::new(ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(module).expect("differentiate pass failed");
    module.as_operation().to_string()
}

// ── Op construction ───────────────────────────────────────────────────────────

#[test]
fn activity_attributes_cover_all_variants() {
    let ctx = create_context();
    let attr = activity_array_attribute(
        &ctx,
        &[
            Activity::Active,
            Activity::Dup,
            Activity::Const,
            Activity::DupNoNeed,
            Activity::ActiveNoNeed,
            Activity::ConstNoNeed,
        ],
    );
    assert_eq!(
        attr.to_string(),
        "[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_dup>, \
         #enzyme<activity enzyme_const>, #enzyme<activity enzyme_dupnoneed>, \
         #enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>]"
    );
}

#[test]
fn autodiff_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        autodiff(
            &ctx,
            "square",
            args,
            &[f64_ty],
            &[Activity::Active],
            &[Activity::ActiveNoNeed],
            1,
            false,
            Location::unknown(&ctx),
        )
    });
    let text = module.as_operation().to_string();
    filecheck(&text, SQUARE, Some("AUTODIFF"));
}

#[test]
fn fwddiff_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        fwddiff(
            &ctx,
            "square",
            args,
            &[f64_ty],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            1,
            false,
            Location::unknown(&ctx),
        )
    });
    let text = module.as_operation().to_string();
    filecheck(&text, SQUARE, Some("FWDDIFF"));
}

#[test]
fn jacobian_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        jacobian(
            &ctx,
            "square",
            args,
            &[f64_ty],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            2,
            true,
            Location::unknown(&ctx),
        )
    });
    let text = module.as_operation().to_string();
    filecheck(&text, SQUARE, Some("JACOBIAN"));
}

#[test]
fn batch_op() {
    let ctx = create_context();
    let tensor_ty = Type::parse(&ctx, "tensor<4xf64>").unwrap();
    let module = with_caller(&ctx, SQUARE, &[tensor_ty], &[tensor_ty], |args| {
        batch(
            &ctx,
            "square",
            args,
            &[tensor_ty],
            &[4],
            Location::unknown(&ctx),
        )
    });
    let text = module.as_operation().to_string();
    filecheck(&text, SQUARE, Some("BATCH"));
}

// ── Differentiate pass ────────────────────────────────────────────────────────

#[test]
fn differentiate_pass_lowers_autodiff() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let mut module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        autodiff(
            &ctx,
            "square",
            args,
            &[f64_ty],
            &[Activity::Active],
            &[Activity::ActiveNoNeed],
            1,
            false,
            Location::unknown(&ctx),
        )
    });
    let text = differentiate(&ctx, &mut module);
    filecheck(&text, SQUARE, Some("LOWERED"));
}

#[test]
fn differentiate_pass_lowers_tensor_autodiff() {
    let ctx = create_context();
    let tensor_ty = Type::parse(&ctx, "tensor<2xf64>").unwrap();
    let mut module = with_caller(
        &ctx,
        SQUARE_TENSOR,
        &[tensor_ty, tensor_ty],
        &[tensor_ty],
        |args| {
            autodiff(
                &ctx,
                "square_tensor",
                args,
                &[tensor_ty],
                &[Activity::Active],
                &[Activity::ActiveNoNeed],
                1,
                false,
                Location::unknown(&ctx),
            )
        },
    );
    let text = differentiate(&ctx, &mut module);
    filecheck(&text, SQUARE_TENSOR, None);
}

// enzyme.jacobian op exists in the dialect but no pass currently lowers it.
#[test]
#[ignore]
fn differentiate_pass_lowers_jacobian() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let mut module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        jacobian(
            &ctx,
            "square",
            args,
            &[f64_ty],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            1,
            false,
            Location::unknown(&ctx),
        )
    });
    let text = differentiate(&ctx, &mut module);
    filecheck(&text, SQUARE, Some("JACOBIAN_LOWERED"));
}
