#![allow(dead_code)]

use melior::{
    dialect::DialectRegistry,
    ir::{
        attribute::{
            ArrayAttribute, BoolAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type,
        TypeLike,
    },
    utility::register_all_dialects,
    Context,
};
use melior_autodiff::{
    activity_attr, enzymeRegisterDialectExtensions, enzyme_dialect_handle, Activity,
};
use mlir_sys::{mlirDialectHandleLoadDialect, mlirFunctionTypeGet};

pub fn setup_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    // Register Enzyme's external AD models for arith, func, scf, etc.
    // before the context is created so extensions fire as each dialect
    // (including BuiltinDialect) is loaded for the first time.
    unsafe { enzymeRegisterDialectExtensions(registry.to_raw()) };
    let ctx = Context::new_with_registry(&registry, false);
    ctx.load_all_available_dialects();
    unsafe {
        let handle = enzyme_dialect_handle();
        mlirDialectHandleLoadDialect(handle, ctx.to_raw());
    }
    ctx
}

pub fn activity_array_attr<'c>(ctx: &'c Context, activities: &[Activity]) -> ArrayAttribute<'c> {
    let raw_ctx = ctx.to_raw();
    let attrs = activities
        .iter()
        .map(|activity| unsafe { Attribute::from_raw(activity_attr(raw_ctx, *activity)) })
        .collect::<Vec<_>>();
    ArrayAttribute::new(ctx, &attrs)
}

pub fn gen_autodiff<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    wrapper_name: &str,
    target_name: &str,
    arg_types: &[&str],
    result_types: &[&str],
    activity: &[Activity],
    ret_activity: &[Activity],
    emit_c_interface: bool,
) {
    append_enzyme_wrapper(
        ctx,
        module,
        "enzyme.autodiff",
        wrapper_name,
        target_name,
        arg_types,
        result_types,
        activity,
        ret_activity,
        emit_c_interface,
    );
}

pub fn gen_fwddiff<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    wrapper_name: &str,
    target_name: &str,
    arg_types: &[&str],
    result_types: &[&str],
    activity: &[Activity],
    ret_activity: &[Activity],
    emit_c_interface: bool,
) {
    append_enzyme_wrapper(
        ctx,
        module,
        "enzyme.fwddiff",
        wrapper_name,
        target_name,
        arg_types,
        result_types,
        activity,
        ret_activity,
        emit_c_interface,
    );
}

pub fn gen_jacobian<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    wrapper_name: &str,
    target_name: &str,
    arg_types: &[&str],
    result_types: &[&str],
    activity: &[Activity],
    ret_activity: &[Activity],
    emit_c_interface: bool,
) {
    append_enzyme_wrapper(
        ctx,
        module,
        "enzyme.jacobian",
        wrapper_name,
        target_name,
        arg_types,
        result_types,
        activity,
        ret_activity,
        emit_c_interface,
    );
}

pub fn gen_batch<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    wrapper_name: &str,
    target_name: &str,
    arg_types: &[&str],
    result_types: &[&str],
    batch_shape: &[i64],
) {
    let loc = Location::unknown(ctx);
    let arg_types = parse_types(ctx, arg_types);
    let result_types = parse_types(ctx, result_types);
    let entry = block_with_args(loc, &arg_types);
    let operands = (0..arg_types.len())
        .map(|index| entry.argument(index).unwrap().into())
        .collect::<Vec<_>>();

    let batch_op = OperationBuilder::new("enzyme.batch", loc)
        .add_operands(&operands)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(ctx, "fn"),
                FlatSymbolRefAttribute::new(ctx, target_name).into(),
            ),
            (
                Identifier::new(ctx, "batch_shape"),
                DenseI64ArrayAttribute::new(ctx, batch_shape).into(),
            ),
        ])
        .build()
        .unwrap();
    let batch_op = entry.append_operation(batch_op);
    let results = (0..result_types.len())
        .map(|index| batch_op.result(index).unwrap().into())
        .collect::<Vec<_>>();

    let ret = OperationBuilder::new("func.return", loc)
        .add_operands(&results)
        .build()
        .unwrap();
    entry.append_operation(ret);
    append_func(
        ctx,
        module,
        loc,
        wrapper_name,
        &arg_types,
        &result_types,
        entry,
        false,
    );
}

fn append_enzyme_wrapper<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    op_name: &str,
    wrapper_name: &str,
    target_name: &str,
    arg_types: &[&str],
    result_types: &[&str],
    activity: &[Activity],
    ret_activity: &[Activity],
    emit_c_interface: bool,
) {
    let loc = Location::unknown(ctx);
    let arg_types = parse_types(ctx, arg_types);
    let result_types = parse_types(ctx, result_types);
    let entry = block_with_args(loc, &arg_types);
    let operands = (0..arg_types.len())
        .map(|index| entry.argument(index).unwrap().into())
        .collect::<Vec<_>>();

    let diff_op = OperationBuilder::new(op_name, loc)
        .add_operands(&operands)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(ctx, "fn"),
                FlatSymbolRefAttribute::new(ctx, target_name).into(),
            ),
            (
                Identifier::new(ctx, "activity"),
                activity_array_attr(ctx, activity).into(),
            ),
            (
                Identifier::new(ctx, "ret_activity"),
                activity_array_attr(ctx, ret_activity).into(),
            ),
            (
                Identifier::new(ctx, "width"),
                IntegerAttribute::new(IntegerType::new(ctx, 64).into(), 1).into(),
            ),
            (
                Identifier::new(ctx, "strong_zero"),
                BoolAttribute::new(ctx, false).into(),
            ),
        ])
        .build()
        .unwrap();
    let diff_op = entry.append_operation(diff_op);
    let results = (0..result_types.len())
        .map(|index| diff_op.result(index).unwrap().into())
        .collect::<Vec<_>>();

    let ret = OperationBuilder::new("func.return", loc)
        .add_operands(&results)
        .build()
        .unwrap();
    entry.append_operation(ret);
    append_func(
        ctx,
        module,
        loc,
        wrapper_name,
        &arg_types,
        &result_types,
        entry,
        emit_c_interface,
    );
}

fn parse_types<'c>(ctx: &'c Context, types: &[&str]) -> Vec<Type<'c>> {
    types
        .iter()
        .map(|ty| Type::parse(ctx, ty).unwrap_or_else(|| panic!("failed to parse type {ty}")))
        .collect()
}

fn block_with_args<'c>(loc: Location<'c>, arg_types: &[Type<'c>]) -> Block<'c> {
    let args = arg_types
        .iter()
        .copied()
        .map(|ty| (ty, loc))
        .collect::<Vec<_>>();
    Block::new(&args)
}

fn append_func<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    loc: Location<'c>,
    name: &str,
    arg_types: &[Type<'c>],
    result_types: &[Type<'c>],
    entry: Block<'c>,
    emit_c_interface: bool,
) {
    let body = Region::new();
    body.append_block(entry);

    let raw_arg_types = arg_types.iter().map(TypeLike::to_raw).collect::<Vec<_>>();
    let raw_result_types = result_types
        .iter()
        .map(TypeLike::to_raw)
        .collect::<Vec<_>>();
    let fn_type = unsafe {
        Type::from_raw(mlirFunctionTypeGet(
            ctx.to_raw(),
            raw_arg_types.len() as isize,
            raw_arg_types.as_ptr(),
            raw_result_types.len() as isize,
            raw_result_types.as_ptr(),
        ))
    };

    let mut attrs = vec![
        (
            Identifier::new(ctx, "sym_name"),
            StringAttribute::new(ctx, name).into(),
        ),
        (
            Identifier::new(ctx, "function_type"),
            TypeAttribute::new(fn_type).into(),
        ),
    ];
    if emit_c_interface {
        attrs.push((
            Identifier::new(ctx, "llvm.emit_c_interface"),
            Attribute::unit(ctx),
        ));
    }

    let func = OperationBuilder::new("func.func", loc)
        .add_attributes(&attrs)
        .add_regions([body])
        .build()
        .unwrap();
    module.body().append_operation(func);
}
