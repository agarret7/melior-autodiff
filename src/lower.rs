//! Lowerings for Enzyme ops that Enzyme emits but provides no upstream-MLIR lowering for.

use melior::{
    dialect::{arith, ods::tensor, DialectHandle},
    ir::{
        attribute::{DenseI32ArrayAttribute, FloatAttribute, IntegerAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{MemRefType, RankedTensorType, ShapedTypeLike, TypeId},
        Attribute, Block, BlockLike, Identifier, Location, OperationRef, Region, RegionLike, Type,
        TypeLike, Value, ValueLike,
    },
    pass::{create_external, ExternalPass, Pass},
    apply_patterns_and_fold_greedily_with_op, create_op_rewrite_pattern, Context,
    GreedyRewriteDriverConfig, PatternRewriter, RewritePattern, RewritePatternSet,
};

#[repr(align(8))]
struct PassId;
static LOWER_ENZYME_HELPERS: PassId = PassId;

/// Lowers `enzyme.fill_zero` to `linalg.fill` and scalar `enzyme.broadcast` to `tensor.splat`.
/// Run before `convert-enzyme-to-memref`, which rejects any enzyme op it does not handle.
pub fn create_lower_enzyme_helpers_pass() -> Pass {
    create_external(
        |op: OperationRef, pass: ExternalPass| {
            let ctx = unsafe { op.context().to_ref() };
            let patterns = RewritePatternSet::new(ctx);
            patterns.add(fill_zero_pattern(ctx));
            patterns.add(broadcast_pattern(ctx));
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
        TypeId::create(&LOWER_ENZYME_HELPERS),
        "LowerEnzymeHelpers",
        "lower-enzyme-helpers",
        "Lower enzyme.fill_zero and enzyme.broadcast to upstream MLIR",
        "",
        &[DialectHandle::arith(), DialectHandle::linalg(), DialectHandle::tensor()],
    )
}

// enzyme.fill_zero %m : memref<...xT>  ->  linalg.fill ins(%zero : T) outs(%m)
fn fill_zero_pattern(ctx: &Context) -> RewritePattern {
    create_op_rewrite_pattern(
        "enzyme.fill_zero",
        1,
        ctx,
        |_, op, rewriter| {
            let op = unsafe { OperationRef::from_raw(op) };
            let rewriter = unsafe { PatternRewriter::from_raw(rewriter) };
            let rewriter = rewriter.as_rewriter_base();
            let ctx = unsafe { op.context().to_ref() };
            let loc = op.location();

            let memref = op.operand(0).unwrap();
            let Ok(memref_type) = MemRefType::try_from(memref.r#type()) else {
                return false;
            };
            let element = memref_type.element();
            let Some(zero) = zero_attribute(ctx, element) else {
                return false;
            };

            rewriter.set_insertion_point_before(op);
            let zero = rewriter.insert(arith::constant(ctx, zero, loc));
            rewriter.insert(linalg_fill(ctx, zero.result(0).unwrap().into(), memref, loc));
            rewriter.erase_op(op);
            true
        },
        &["arith.constant", "linalg.fill"],
    )
}

// enzyme.broadcast %x {shape} : T -> tensor<...xT>  ->  tensor.splat %x
// A rank-0 tensor %x is extracted to a scalar first; higher-rank inputs are not handled yet.
fn broadcast_pattern(ctx: &Context) -> RewritePattern {
    create_op_rewrite_pattern(
        "enzyme.broadcast",
        1,
        ctx,
        |_, op, rewriter| {
            let op = unsafe { OperationRef::from_raw(op) };
            let rewriter = unsafe { PatternRewriter::from_raw(rewriter) };
            let rewriter = rewriter.as_rewriter_base();
            let ctx = unsafe { op.context().to_ref() };

            let loc = op.location();
            let input = op.operand(0).unwrap();
            let result_type = op.result(0).unwrap().r#type();
            rewriter.set_insertion_point_before(op);

            let input: Value = if input.r#type().is_shaped() {
                let Ok(input_type) = RankedTensorType::try_from(input.r#type()) else {
                    return false;
                };
                if input_type.rank() != 0 {
                    return false;
                }
                rewriter
                    .insert(tensor::extract(ctx, input_type.element(), input, &[], loc).into())
                    .result(0)
                    .unwrap()
                    .into()
            } else {
                input
            };
            let splat = rewriter.insert(
                tensor::splat(ctx, result_type, input, &[], op.location()).into(),
            );
            rewriter.replace_op_with_operation(op, splat);
            true
        },
        &["tensor.extract", "tensor.splat"],
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

// melior has no builder for linalg structured ops, so the body region is built by hand.
// A tensor `output` yields the filled tensor; a memref `output` is filled in place.
pub(crate) fn linalg_fill<'c>(
    ctx: &'c Context,
    value: Value<'c, '_>,
    output: Value<'c, '_>,
    loc: Location<'c>,
) -> melior::ir::Operation<'c> {
    let element = value.r#type();
    let body = Block::new(&[(element, loc), (element, loc)]);
    let input = body.argument(0).unwrap().into();
    body.append_operation(
        OperationBuilder::new("linalg.yield", loc)
            .add_operands(&[input])
            .build()
            .unwrap(),
    );
    let region = Region::new();
    region.append_block(body);

    let results: Vec<Type> = if output.r#type().is_tensor() {
        vec![output.r#type()]
    } else {
        vec![]
    };
    OperationBuilder::new("linalg.fill", loc)
        .add_operands(&[value, output])
        .add_results(&results)
        .add_attributes(&[(
            Identifier::new(ctx, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(ctx, &[1, 1]).into(),
        )])
        .add_regions([region])
        .build()
        .unwrap()
}
