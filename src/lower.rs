//! Lowerings for Enzyme ops that Enzyme generates but has no pass to lower.

use melior::{
    dialect::{
        arith,
        ods::{linalg, tensor},
        DialectHandle,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, FloatAttribute, IntegerAttribute},
        operation::OperationLike,
        r#type::{MemRefType, RankedTensorType, ShapedTypeLike, TypeId},
        Attribute, Location, Operation, OperationRef, Region, Type, TypeLike, Value, ValueLike,
    },
    pass::{create_external, ExternalPass, Pass},
    apply_patterns_and_fold_greedily_with_op, create_op_rewrite_pattern, Context,
    Error, GreedyRewriteDriverConfig, RewritePattern, RewritePatternSet,
};

#[repr(align(8))]
struct PassId;
static LOWER_FILL_ZERO: PassId = PassId;
static LOWER_BROADCAST: PassId = PassId;

/// `lower-fill-zero`: lowers `enzyme.fill_zero` to `linalg.fill` with zero.
///
/// Enzyme emits `enzyme.fill_zero` to zero a memref's shadow in reverse mode, for example for a
/// `memref.alloca`, but has no pass to lower it, and `convert-enzyme-to-memref` fails on it. Run
/// this before `convert-enzyme-to-memref`.
pub fn create_lower_fill_zero_pass() -> Pass {
    pattern_pass(
        fill_zero_pattern,
        TypeId::create(&LOWER_FILL_ZERO),
        "LowerFillZero",
        "lower-fill-zero",
        "Lower enzyme.fill_zero to linalg.fill",
        &[DialectHandle::arith(), DialectHandle::linalg()],
    )
}

/// `lower-broadcast`: lowers `enzyme.broadcast` to `tensor.splat` (scalar input) or
/// `linalg.broadcast` (tensor input).
///
/// Enzyme emits `enzyme.broadcast` when it differentiates with `width` above one, to give each
/// tangent its own copy of a value, but has no pass to lower it. Run this after differentiating.
pub fn create_lower_broadcast_pass() -> Pass {
    pattern_pass(
        broadcast_pattern,
        TypeId::create(&LOWER_BROADCAST),
        "LowerBroadcast",
        "lower-broadcast",
        "Lower enzyme.broadcast to tensor.splat or linalg.broadcast",
        &[DialectHandle::linalg(), DialectHandle::tensor()],
    )
}

// A pass that applies one rewrite pattern greedily.
fn pattern_pass(
    pattern: fn(&Context) -> RewritePattern,
    id: TypeId,
    name: &str,
    argument: &str,
    description: &str,
    dependent_dialects: &[DialectHandle],
) -> Pass {
    create_external(
        move |op: OperationRef, pass: ExternalPass| {
            let context = op.context();
            let ctx = unsafe { context.to_ref() };
            let patterns = RewritePatternSet::new(ctx);
            patterns.add(pattern(ctx));
            if apply_patterns_and_fold_greedily_with_op(
                op,
                patterns.freeze(),
                &GreedyRewriteDriverConfig::new(),
            )
            .is_err()
            {
                pass.signal_failure();
            }
        },
        id,
        name,
        argument,
        description,
        "",
        dependent_dialects,
    )
}

// enzyme.fill_zero %m : memref<...xT>  ->  linalg.fill ins(%zero : T) outs(%m)
fn fill_zero_pattern(ctx: &Context) -> RewritePattern {
    create_op_rewrite_pattern(
        "enzyme.fill_zero",
        1,
        ctx,
        |op, rewriter| {
            let rewriter = rewriter.as_rewriter_base();
            let context = op.context();
            let ctx = unsafe { context.to_ref() };
            let loc = op.location();

            let memref = op.operand(0).unwrap();
            let memref_type = MemRefType::try_from(memref.r#type())
                .map_err(|_| Error::TypeExpected("memref", memref.r#type().to_string()))?;
            let element = memref_type.element();
            let zero = zero_attribute(ctx, element)
                .ok_or_else(|| Error::TypeExpected("integer or float", element.to_string()))?;

            rewriter.set_insertion_point_before(op);
            let zero = rewriter.insert(arith::constant(ctx, zero, loc));
            rewriter.insert(linalg_fill(ctx, zero.result(0).unwrap().into(), memref, loc));
            rewriter.erase_op(op);
            Ok(())
        },
        &["arith.constant", "linalg.fill"],
    )
}

// enzyme.broadcast adds the leading dims in its `shape` attribute:
//   scalar %x : T             ->  tensor.splat %x : tensor<shape x T>
//   tensor %x : tensor<D x T> ->  linalg.broadcast ins(%x) outs(tensor.empty : tensor<shape x D x T>)
fn broadcast_pattern(ctx: &Context) -> RewritePattern {
    create_op_rewrite_pattern(
        "enzyme.broadcast",
        1,
        ctx,
        |op, rewriter| {
            let rewriter = rewriter.as_rewriter_base();
            let context = op.context();
            let ctx = unsafe { context.to_ref() };

            let loc = op.location();
            let input = op.operand(0).unwrap();
            let result_type = RankedTensorType::try_from(op.result(0).unwrap().r#type())
                .map_err(|_| Error::TypeExpected("ranked tensor", op.result(0).unwrap().r#type().to_string()))?;
            rewriter.set_insertion_point_before(op);

            if !input.r#type().is_shaped() {
                let splat = rewriter.insert(tensor::splat(ctx, result_type.into(), input, &[], loc).into());
                rewriter.replace_op_with_operation(op, splat);
                return Ok(());
            }

            let input_type = RankedTensorType::try_from(input.r#type())
                .map_err(|_| Error::TypeExpected("ranked tensor", input.r#type().to_string()))?;
            if !result_type.has_static_shape() {
                return Err(Error::TypeExpected("statically shaped tensor", result_type.to_string()));
            }
            // The added dims are the leading ones.
            let added = (result_type.rank() - input_type.rank()) as i64;
            let dimensions: Vec<i64> = (0..added).collect();

            let empty = rewriter.insert(tensor::empty(ctx, result_type.into(), &[], loc).into());
            let broadcast = linalg_named(
                linalg::broadcast(
                    ctx,
                    &[result_type.into()],
                    input,
                    empty.result(0).unwrap().into(),
                    Region::new(),
                    DenseI64ArrayAttribute::new(ctx, &dimensions),
                    loc,
                )
                .into(),
            );
            let broadcast = rewriter.insert(broadcast);
            rewriter.replace_op_with_operation(op, broadcast);
            Ok(())
        },
        &["tensor.splat", "tensor.empty", "linalg.broadcast"],
    )
}

pub(crate) fn zero_attribute<'c>(ctx: &'c Context, element: Type<'c>) -> Option<Attribute<'c>> {
    if element.is_float() {
        Some(FloatAttribute::new(ctx, element, 0.0).into())
    } else if element.is_integer() || element.is_index() {
        Some(IntegerAttribute::new(element, 0).into())
    } else {
        None
    }
}

// A tensor `output` yields the filled tensor; a memref `output` is filled in place.
pub(crate) fn linalg_fill<'c>(
    ctx: &'c Context,
    value: Value<'c, '_>,
    output: Value<'c, '_>,
    loc: Location<'c>,
) -> melior::ir::Operation<'c> {
    let results: Vec<Type> = if output.r#type().is_tensor() {
        vec![output.r#type()]
    } else {
        vec![]
    };
    linalg_named(linalg::fill(ctx, &results, &[value], &[output], Region::new(), loc).into())
}

// Named linalg ops carry their body as a region even though the op fixes it; MLIR builds it.
fn linalg_named(op: Operation) -> Operation {
    unsafe { mlir_sys::mlirLinalgFillBuiltinNamedOpRegion(op.to_raw()) };
    op
}
