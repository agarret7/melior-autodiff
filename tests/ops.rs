use melior::dialect::func;
use melior::ir::{
    attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
    operation::OperationLike,
    r#type::FunctionType,
    Block, BlockLike, Location, Module, Operation, Region, RegionLike, Type, Value,
};
use melior::pass::PassManager;
use melior::Context;
use melior_autodiff::{
    activity_array_attribute, autodiff, batch, fwddiff, jacobian, ods, pass,
    utility::create_context,
    Activity,
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
    pm.add_pass(pass::create_differentiate_pass());
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


fn symbol<'c>(ctx: &'c Context, name: &str) -> FlatSymbolRefAttribute<'c> {
    FlatSymbolRefAttribute::new(ctx, name)
}

// `enzyme.autodiff` of `square`, active in x and in its result.
fn autodiff_square<'c>(
    ctx: &'c Context,
    args: &[Value<'c, '_>],
    width: i64,
    strong_zero: bool,
) -> Operation<'c> {
    let f64_ty = Type::float64(ctx);
    autodiff(
        ctx,
        "square",
        args,
        &[f64_ty],
        &[Activity::Active],
        &[Activity::ActiveNoNeed],
        width,
        strong_zero,
        Location::unknown(ctx),
    )
}

#[test]
fn autodiff_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        autodiff_square(&ctx, args, 1, false)
    });
    filecheck(&module.as_operation().to_string(), SQUARE, Some("AUTODIFF"));
}

// The generated builder that `autodiff` wraps makes the same op.
#[test]
fn ods_autodiff_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        ods::autodiff(
            &ctx,
            &[f64_ty],
            args,
            symbol(&ctx, "square"),
            activity_array_attribute(&ctx, &[Activity::Active]),
            activity_array_attribute(&ctx, &[Activity::ActiveNoNeed]),
            Location::unknown(&ctx),
        )
        .into()
    });
    filecheck(&module.as_operation().to_string(), SQUARE, Some("AUTODIFF"));
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
    filecheck(&module.as_operation().to_string(), SQUARE, Some("FWDDIFF"));
}

// `enzyme.jacobian` of `square`.
fn jacobian_square<'c>(
    ctx: &'c Context,
    args: &[Value<'c, '_>],
    width: i64,
    strong_zero: bool,
) -> Operation<'c> {
    let f64_ty = Type::float64(ctx);
    jacobian(
        ctx,
        "square",
        args,
        &[f64_ty],
        &[Activity::Dup],
        &[Activity::DupNoNeed],
        width,
        strong_zero,
        Location::unknown(ctx),
    )
}

#[test]
fn jacobian_op() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        jacobian_square(&ctx, args, 2, true)
    });
    filecheck(&module.as_operation().to_string(), SQUARE, Some("JACOBIAN"));
}

#[test]
fn batch_op() {
    let ctx = create_context();
    let tensor_ty = Type::parse(&ctx, "tensor<4xf64>").unwrap();
    let module = with_caller(&ctx, SQUARE, &[tensor_ty], &[tensor_ty], |args| {
        batch(&ctx, "square", args, &[tensor_ty], &[4], Location::unknown(&ctx))
    });
    filecheck(&module.as_operation().to_string(), SQUARE, Some("BATCH"));
}

// The generated op type reads back the optional attributes `autodiff` set.
#[test]
fn autodiff_accessors() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let loc = Location::unknown(&ctx);
    let block = Block::new(&[(f64_ty, loc), (f64_ty, loc)]);
    let args = [block.argument(0).unwrap().into(), block.argument(1).unwrap().into()];
    let op = autodiff_square(&ctx, &args, 2, true);
    let op = ods::AutoDiffOperation::try_from(op).expect("not an enzyme.autodiff");
    assert_eq!(op.r#fn().unwrap().value(), "square");
    assert_eq!(op.inputs().count(), 2);
    assert_eq!(op.width().unwrap().value(), 2);
    assert!(op.strong_zero().unwrap().value());
}

// ── Differentiate pass ────────────────────────────────────────────────────────

#[test]
fn differentiate_pass_lowers_autodiff() {
    let ctx = create_context();
    let f64_ty = Type::float64(&ctx);
    let mut module = with_caller(&ctx, SQUARE, &[f64_ty, f64_ty], &[f64_ty], |args| {
        autodiff_square(&ctx, args, 1, false)
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
        jacobian_square(&ctx, args, 1, false)
    });
    let text = differentiate(&ctx, &mut module);
    filecheck(&text, SQUARE, Some("JACOBIAN_LOWERED"));
}
