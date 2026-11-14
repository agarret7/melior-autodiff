//! Lowerings for Enzyme ops that Enzyme generates but has no pass to lower.
//!
//! This module is a stopgap, to be removed once Enzyme lowers these ops itself: `enzyme.fill_zero`
//! in `convert-enzyme-to-memref`, and `enzyme.broadcast` in `enzyme-batch-to-tensor`.

use std::{cell::RefCell, collections::HashMap, ffi::CString, rc::Rc};

use melior::{
    dialect::{
        arith,
        ods::{linalg, tensor},
        DialectHandle,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, FloatAttribute, IntegerAttribute},
        operation::{OperationLike, WalkOrder, WalkResult},
        r#type::{MemRefType, RankedTensorType, ShapedTypeLike, TypeId},
        Attribute, Location, Operation, OperationRef, Region, Type, TypeLike, Value, ValueLike,
    },
    pass::{create_external, ExternalPass, Pass},
    apply_patterns_and_fold_greedily_with_op, create_op_rewrite_pattern, Context, Error,
    GreedyRewriteDriverConfig, PatternRewriter, RewritePatternSet,
};

#[repr(align(8))]
struct PassId;
static LOWER_FILL_ZERO: PassId = PassId;
static LOWER_BROADCAST: PassId = PassId;

/// `enzyme-lower-fill-zero`: lowers `enzyme.fill_zero` to `linalg.fill` with zero.
///
/// Enzyme emits `enzyme.fill_zero` to zero a memref's shadow in reverse mode, for example for a
/// `memref.alloca`, but has no pass to lower it, and `convert-enzyme-to-memref` fails on it. Run
/// this before `convert-enzyme-to-memref`.
pub fn create_lower_fill_zero_pass() -> Pass {
    pattern_pass(
        "enzyme.fill_zero",
        lower_fill_zero,
        &["arith.constant", "linalg.fill"],
        TypeId::create(&LOWER_FILL_ZERO),
        "LowerFillZero",
        "enzyme-lower-fill-zero",
        "Lower enzyme.fill_zero to linalg.fill",
        &[DialectHandle::arith(), DialectHandle::linalg()],
    )
}

/// `enzyme-lower-broadcast`: lowers `enzyme.broadcast` to `tensor.splat` (scalar input) or
/// `linalg.broadcast` (tensor input).
///
/// Enzyme emits `enzyme.broadcast` when it differentiates with `width` above one, to give each
/// tangent its own copy of a value, but has no pass to lower it. Run this after differentiating.
pub fn create_lower_broadcast_pass() -> Pass {
    pattern_pass(
        "enzyme.broadcast",
        lower_broadcast,
        &["tensor.splat", "tensor.empty", "linalg.broadcast"],
        TypeId::create(&LOWER_BROADCAST),
        "LowerBroadcast",
        "enzyme-lower-broadcast",
        "Lower enzyme.broadcast to tensor.splat or linalg.broadcast",
        &[DialectHandle::linalg(), DialectHandle::tensor()],
    )
}

type Lowering = for<'c> fn(OperationRef<'c, '_>, &PatternRewriter<'c, '_>) -> Result<(), Error>;

// A pass that rewrites every `op_name` op with `lower`. Any op it can't lower is reported as an
// error at that op, and the pass fails.
#[allow(clippy::too_many_arguments)]
fn pattern_pass(
    op_name: &'static str,
    lower: Lowering,
    generated_names: &'static [&'static str],
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

            // A pattern only reports whether it matched, so keep each op's reason for the report
            // below.
            let reasons = Rc::new(RefCell::new(HashMap::new()));
            let recorded = reasons.clone();
            let pattern = create_op_rewrite_pattern(
                op_name,
                1,
                ctx,
                move |op, rewriter| {
                    let key = op.to_raw().ptr as usize;
                    let result = lower(op, rewriter);
                    match &result {
                        Ok(()) => recorded.borrow_mut().remove(&key),
                        Err(error) => recorded.borrow_mut().insert(key, error.to_string()),
                    };
                    result.is_ok()
                },
                generated_names,
            );
            let patterns = RewritePatternSet::new(ctx);
            patterns.add(pattern);
            // The pass manager hands this pass `op` to transform, so borrowing it mutably for the
            // driver is sound; nothing else holds a handle into it while the pass runs.
            let mut target = unsafe { melior::ir::operation::OperationRefMut::from_raw(op.to_raw()) };
            if apply_patterns_and_fold_greedily_with_op(
                &mut target,
                &patterns.freeze(),
                &GreedyRewriteDriverConfig::new(),
            )
            .is_err()
            {
                pass.signal_failure();
            }

            let mut failed = false;
            op.walk(WalkOrder::PreOrder, |nested| {
                if nested.name().as_string_ref().as_str() == Ok(op_name) {
                    let reasons = reasons.borrow();
                    let reason = reasons
                        .get(&(nested.to_raw().ptr as usize))
                        .map_or("no lowering applies", String::as_str);
                    emit_error(nested.location(), &format!("cannot lower {op_name}: {reason}"));
                    failed = true;
                }
                WalkResult::Advance
            });
            if failed {
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

fn emit_error(location: Location, message: &str) {
    let message = CString::new(message.replace('\0', " ")).unwrap();
    unsafe { mlir_sys::mlirEmitError(location.to_raw(), message.as_ptr()) };
}

// enzyme.fill_zero %m : memref<...xT>  ->  linalg.fill ins(%zero : T) outs(%m)
fn lower_fill_zero<'c>(op: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> Result<(), Error> {
    let rewriter = rewriter.as_rewriter_base();
    let context = op.context();
    let ctx = unsafe { context.to_ref() };
    let loc = op.location();

    let memref = op.operand(0)?;
    let memref_type = MemRefType::try_from(memref.r#type())
        .map_err(|_| Error::TypeExpected("memref", memref.r#type().to_string()))?;
    let element = memref_type.element();
    let zero = zero_attribute(ctx, element)
        .ok_or_else(|| Error::TypeExpected("integer or float", element.to_string()))?;

    rewriter.set_insertion_point_before(op);
    let zero = rewriter.insert(arith::constant(ctx, zero, loc));
    rewriter.insert(linalg_fill(ctx, zero.result(0)?.into(), memref, loc));
    rewriter.erase_op(op);
    Ok(())
}

// enzyme.broadcast adds the leading dims in its `shape` attribute:
//   scalar %x : T             ->  tensor.splat %x : tensor<shape x T>
//   tensor %x : tensor<D x T> ->  linalg.broadcast ins(%x) outs(tensor.empty : tensor<shape x D x T>)
fn lower_broadcast<'c>(op: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> Result<(), Error> {
    let rewriter = rewriter.as_rewriter_base();
    let context = op.context();
    let ctx = unsafe { context.to_ref() };

    let loc = op.location();
    let input = op.operand(0)?;
    let result = op.result(0)?.r#type();
    let result_type = RankedTensorType::try_from(result)
        .map_err(|_| Error::TypeExpected("ranked tensor", result.to_string()))?;

    if !input.r#type().is_shaped() {
        rewriter.set_insertion_point_before(op);
        let splat = rewriter.insert(tensor::splat(ctx, result_type.into(), input, &[], loc).into());
        rewriter.replace_op_with_operation(op, splat);
        return Ok(());
    }

    let input_type = RankedTensorType::try_from(input.r#type())
        .map_err(|_| Error::TypeExpected("ranked tensor", input.r#type().to_string()))?;
    // The added dims are the leading ones.
    let added = result_type.rank() - input_type.rank();
    let dimensions: Vec<i64> = (0..added as i64).collect();

    // `shape` fixes the added dims; each dynamic dim after them takes its size from the input.
    let mut dynamic = Vec::new();
    for dim in 0..result_type.rank() {
        // melior's `dim_size` returns a `DimSize` it doesn't export, so ask MLIR directly.
        if unsafe { mlir_sys::mlirShapedTypeIsDynamicDim(result_type.to_raw(), dim as isize) } {
            if dim < added {
                return Err(Error::TypeExpected(
                    "tensor with static leading (batch) dims",
                    result_type.to_string(),
                ));
            }
            dynamic.push(dim - added);
        }
    }

    rewriter.set_insertion_point_before(op);
    let mut sizes = Vec::new();
    for dim in dynamic {
        let index = rewriter.insert(arith::constant(
            ctx,
            IntegerAttribute::new(Type::index(ctx), dim as i64).into(),
            loc,
        ));
        let size = rewriter.insert(
            tensor::dim(ctx, Type::index(ctx), input, index.result(0)?.into(), loc).into(),
        );
        sizes.push(size.result(0)?.into());
    }
    let empty = rewriter.insert(tensor::empty(ctx, result_type.into(), &sizes, loc).into());
    let broadcast = linalg_named(
        linalg::broadcast(
            ctx,
            &[result_type.into()],
            input,
            empty.result(0)?.into(),
            Region::new(),
            DenseI64ArrayAttribute::new(ctx, &dimensions),
            loc,
        )
        .into(),
    );
    let broadcast = rewriter.insert(broadcast);
    rewriter.replace_op_with_operation(op, broadcast);
    Ok(())
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
