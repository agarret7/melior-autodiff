//! Lowers the Impulse ops left after `expand-impulse` and differentiation to upstream
//! tensor/linalg/scf/arith/math, following the semantics of Enzyme-JAX's StableHLO lowering.

use melior::{
    dialect::{arith, ods, DialectHandle},
    ir::{
        attribute::{
            AttributeLike, DenseI32ArrayAttribute, DenseI64ArrayAttribute, FloatAttribute,
            IntegerAttribute,
        },
        operation::{OperationBuilder, OperationLike},
        r#type::{IntegerType, RankedTensorType, TypeId},
        Attribute, Block, BlockLike, Identifier, Location, Operation, OperationRef, Region,
        RegionLike, Type, TypeLike, Value, ValueLike,
    },
    pass::{create_external, ExternalPass, Pass},
    apply_patterns_and_fold_greedily_with_op, create_op_rewrite_pattern, Context,
    GreedyRewriteDriverConfig, PatternRewriter, RewritePattern, RewritePatternSet, RewriterBase,
};

use crate::lower::{linalg_fill, zero_attribute};

#[repr(align(8))]
struct PassId;
static LOWER_IMPULSE: PassId = PassId;

/// Lowers `impulse.{select, reshape, slice, dynamic_slice, dynamic_update_slice, dot, for, if,
/// randomSplit, random}` to upstream MLIR. Run after `enzyme` (differentiation).
pub fn create_lower_impulse_pass() -> Pass {
    create_external(
        |op: OperationRef, pass: ExternalPass| {
            let ctx = unsafe { op.context().to_ref() };
            let patterns = RewritePatternSet::new(ctx);
            patterns.add(pattern(ctx, "impulse.select", lower_select));
            patterns.add(pattern(ctx, "impulse.reshape", lower_reshape));
            patterns.add(pattern(ctx, "impulse.slice", lower_slice));
            patterns.add(pattern(ctx, "impulse.dynamic_slice", lower_dynamic_slice));
            patterns.add(pattern(ctx, "impulse.dynamic_update_slice", lower_dynamic_update_slice));
            patterns.add(pattern(ctx, "impulse.dot", lower_dot));
            patterns.add(pattern(ctx, "impulse.for", lower_for));
            patterns.add(pattern(ctx, "impulse.if", lower_if));
            patterns.add(pattern(ctx, "impulse.randomSplit", lower_random_split));
            patterns.add(pattern(ctx, "impulse.random", lower_random));
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
        TypeId::create(&LOWER_IMPULSE),
        "LowerImpulse",
        "lower-impulse",
        "Lower Impulse ops to upstream tensor/linalg/scf/arith/math",
        "",
        &[
            DialectHandle::arith(),
            DialectHandle::linalg(),
            DialectHandle::scf(),
            DialectHandle::tensor(),
        ],
    )
}

type Lowering = for<'c, 'a, 'r> fn(&'c Context, OperationRef<'c, 'a>, &RewriterBase<'c, 'r>) -> bool;

fn pattern(ctx: &Context, root: &str, lower: Lowering) -> RewritePattern {
    create_op_rewrite_pattern(
        root,
        1,
        ctx,
        move |_, op, rewriter| {
            let op = unsafe { OperationRef::from_raw(op) };
            let rewriter = unsafe { PatternRewriter::from_raw(rewriter) };
            let rewriter = rewriter.as_rewriter_base();
            let ctx = unsafe { op.context().to_ref() };
            rewriter.set_insertion_point_before(op);
            lower(ctx, op, &rewriter)
        },
        &[],
    )
}

// ── Building blocks ───────────────────────────────────────────────────────────

/// Somewhere ops can be emitted: the rewriter's insertion point, or a detached block.
trait Emit<'c> {
    fn emit<'s>(&'s self, op: Operation<'c>) -> OperationRef<'c, 's>;

    fn value<'s>(&'s self, op: Operation<'c>) -> Value<'c, 's>
    where
        'c: 's,
    {
        self.emit(op).result(0).unwrap().into()
    }
}

impl<'c> Emit<'c> for RewriterBase<'c, '_> {
    fn emit<'s>(&'s self, op: Operation<'c>) -> OperationRef<'c, 's> {
        self.insert(op)
    }
}

impl<'c> Emit<'c> for Block<'c> {
    fn emit<'s>(&'s self, op: Operation<'c>) -> OperationRef<'c, 's> {
        self.append_operation(op)
    }
}

fn build<'c>(name: &str, operands: &[Value<'c, '_>], result: Type<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new(name, loc)
        .add_operands(operands)
        .add_results(&[result])
        .build()
        .unwrap()
}

fn i64_type(ctx: &Context) -> Type<'_> {
    IntegerType::new(ctx, 64).into()
}

fn const_i64<'c>(ctx: &'c Context, value: i64, loc: Location<'c>) -> Operation<'c> {
    arith::constant(ctx, IntegerAttribute::new(i64_type(ctx), value).into(), loc)
}

fn const_index<'c>(ctx: &'c Context, value: i64, loc: Location<'c>) -> Operation<'c> {
    arith::constant(ctx, IntegerAttribute::new(Type::index(ctx), value).into(), loc)
}

fn const_f64<'c>(ctx: &'c Context, value: f64, loc: Location<'c>) -> Operation<'c> {
    arith::constant(ctx, FloatAttribute::new(ctx, Type::float64(ctx), value).into(), loc)
}

fn shape(ty: Type) -> Vec<i64> {
    unsafe {
        let raw = ty.to_raw();
        (0..mlir_sys::mlirShapedTypeGetRank(raw))
            .map(|i| mlir_sys::mlirShapedTypeGetDimSize(raw, i as isize))
            .collect()
    }
}

fn element_type(ty: Type) -> Type {
    unsafe { Type::from_raw(mlir_sys::mlirShapedTypeGetElementType(ty.to_raw())) }
}

fn i64_array(attr: Attribute) -> Vec<i64> {
    unsafe {
        let raw = attr.to_raw();
        (0..mlir_sys::mlirDenseArrayGetNumElements(raw))
            .map(|i| mlir_sys::mlirDenseI64ArrayGetElement(raw, i))
            .collect()
    }
}

fn is_rank0_tensor(ty: Type) -> bool {
    RankedTensorType::try_from(ty).is_ok_and(|t| shape(t.into()).is_empty())
}

/// Extracts the scalar from a rank-0 tensor; scalars pass through.
fn scalar<'c: 's, 's, E: Emit<'c>>(ctx: &'c Context, e: &'s E, v: Value<'c, 's>, loc: Location<'c>) -> Value<'c, 's> {
    if is_rank0_tensor(v.r#type()) {
        e.value(ods::tensor::extract(ctx, element_type(v.r#type()), v, &[], loc).into())
    } else {
        v
    }
}

/// A `tensor<i64>` index operand as an `index` clamped to `[0, hi]`, as StableHLO clamps slices.
fn clamped_index<'c, 's>(ctx: &'c Context, rw: &'s RewriterBase<'c, '_>, v: Value<'c, 's>, hi: i64, loc: Location<'c>) -> Value<'c, 's> {
    let i = scalar(ctx, rw, v, loc);
    let i = rw.value(build("arith.index_cast", &[i], Type::index(ctx), loc));
    let hi = rw.value(const_index(ctx, hi, loc));
    let zero = rw.value(const_index(ctx, 0, loc));
    let i = rw.value(build("arith.minsi", &[i, hi], Type::index(ctx), loc));
    rw.value(build("arith.maxsi", &[i, zero], Type::index(ctx), loc))
}

fn dense_i64<'c>(ctx: &'c Context, name: &str, values: &[i64]) -> (Identifier<'c>, Attribute<'c>) {
    (Identifier::new(ctx, name), DenseI64ArrayAttribute::new(ctx, values).into())
}

fn segments<'c>(ctx: &'c Context, sizes: &[i32]) -> (Identifier<'c>, Attribute<'c>) {
    (Identifier::new(ctx, "operandSegmentSizes"), DenseI32ArrayAttribute::new(ctx, sizes).into())
}

const DYNAMIC: i64 = i64::MIN;

// ── Lowerings ─────────────────────────────────────────────────────────────────

// impulse.select %c : tensor<i1> broadcasts a scalar condition, like stablehlo.select.
fn lower_select<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let cond = scalar(ctx, rw, op.operand(0).unwrap(), loc);
    let select = rw.emit(arith::select(cond, op.operand(1).unwrap(), op.operand(2).unwrap(), loc));
    rw.replace_op_with_operation(op, select);
    true
}

fn lower_reshape<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let result_type = op.result(0).unwrap().r#type();
    let dims = shape(result_type);
    let text = format!(
        "dense<[{}]> : tensor<{}xindex>",
        dims.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        dims.len()
    );
    let shape_value = rw.value(arith::constant(ctx, Attribute::parse(ctx, &text).unwrap(), loc));
    let reshape = rw.emit(op_builder_reshape(op.operand(0).unwrap(), shape_value, result_type, loc));
    rw.replace_op_with_operation(op, reshape);
    true
}

fn op_builder_reshape<'c>(src: Value<'c, '_>, shape: Value<'c, '_>, result: Type<'c>, loc: Location<'c>) -> Operation<'c> {
    build("tensor.reshape", &[src, shape], result, loc)
}

fn extract_slice<'c>(
    ctx: &'c Context,
    src: Value<'c, '_>,
    dynamic_offsets: &[Value<'c, '_>],
    static_offsets: &[i64],
    sizes: &[i64],
    strides: &[i64],
    result: Type<'c>,
    loc: Location<'c>,
) -> Operation<'c> {
    let mut operands = vec![src];
    operands.extend_from_slice(dynamic_offsets);
    OperationBuilder::new("tensor.extract_slice", loc)
        .add_operands(&operands)
        .add_results(&[result])
        .add_attributes(&[
            dense_i64(ctx, "static_offsets", static_offsets),
            dense_i64(ctx, "static_sizes", sizes),
            dense_i64(ctx, "static_strides", strides),
            segments(ctx, &[1, dynamic_offsets.len() as i32, 0, 0]),
        ])
        .build()
        .unwrap()
}

fn lower_slice<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let attr = |name| i64_array(op.attribute(name).unwrap());
    let (start, limit, strides) = (attr("start_indices"), attr("limit_indices"), attr("strides"));
    let sizes: Vec<i64> = (0..start.len())
        .map(|i| (limit[i] - start[i] + strides[i] - 1) / strides[i])
        .collect();
    let result_type = op.result(0).unwrap().r#type();
    let slice = rw.emit(extract_slice(
        ctx,
        op.operand(0).unwrap(),
        &[],
        &start,
        &sizes,
        &strides,
        result_type,
        op.location(),
    ));
    rw.replace_op_with_operation(op, slice);
    true
}

fn lower_dynamic_slice<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let src = op.operand(0).unwrap();
    let dims = shape(src.r#type());
    let sizes = i64_array(op.attribute("slice_sizes").unwrap());
    let offsets: Vec<Value> = (0..dims.len())
        .map(|i| clamped_index(ctx, rw, op.operand(i + 1).unwrap(), dims[i] - sizes[i], loc))
        .collect();
    let slice = rw.emit(extract_slice(
        ctx,
        src,
        &offsets,
        &vec![DYNAMIC; dims.len()],
        &sizes,
        &vec![1; dims.len()],
        op.result(0).unwrap().r#type(),
        loc,
    ));
    rw.replace_op_with_operation(op, slice);
    true
}

fn lower_dynamic_update_slice<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let dest = op.operand(0).unwrap();
    let update = op.operand(1).unwrap();
    let dims = shape(dest.r#type());
    let sizes = shape(update.r#type());
    let offsets: Vec<Value> = (0..dims.len())
        .map(|i| clamped_index(ctx, rw, op.operand(i + 2).unwrap(), dims[i] - sizes[i], loc))
        .collect();
    let mut operands = vec![update, dest];
    operands.extend(offsets.iter().copied());
    let insert = rw.emit(
        OperationBuilder::new("tensor.insert_slice", loc)
            .add_operands(&operands)
            .add_results(&[dest.r#type()])
            .add_attributes(&[
                dense_i64(ctx, "static_offsets", &vec![DYNAMIC; dims.len()]),
                dense_i64(ctx, "static_sizes", &sizes),
                dense_i64(ctx, "static_strides", &vec![1; dims.len()]),
                segments(ctx, &[1, 1, dims.len() as i32, 0, 0]),
            ])
            .build()
            .unwrap(),
    );
    rw.replace_op_with_operation(op, insert);
    true
}

// impulse.dot is stablehlo.dot_general: result dims are batch, lhs free, rhs free.
fn lower_dot<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let (lhs, rhs) = (op.operand(0).unwrap(), op.operand(1).unwrap());
    let out_type = op.result(0).unwrap().r#type();
    let element = element_type(out_type);
    if !element.is_float() {
        return false;
    }
    let attr = |name| i64_array(op.attribute(name).unwrap());
    let (lb, rb) = (attr("lhs_batching_dimensions"), attr("rhs_batching_dimensions"));
    let (lc, rc) = (attr("lhs_contracting_dimensions"), attr("rhs_contracting_dimensions"));
    let (lrank, rrank) = (shape(lhs.r#type()).len() as i64, shape(rhs.r#type()).len() as i64);
    let lfree: Vec<i64> = (0..lrank).filter(|d| !lb.contains(d) && !lc.contains(d)).collect();
    let rfree: Vec<i64> = (0..rrank).filter(|d| !rb.contains(d) && !rc.contains(d)).collect();
    let (nb, nlf, nrf, nc) = (lb.len(), lfree.len(), rfree.len(), lc.len());
    let total = nb + nlf + nrf + nc;

    let d = |i: usize| format!("d{i}");
    let operand_map = |rank: i64, batch: &[i64], contract: &[i64], free: &[i64], free_base: usize| {
        (0..rank)
            .map(|dim| {
                if let Some(k) = batch.iter().position(|&x| x == dim) {
                    d(k)
                } else if let Some(k) = contract.iter().position(|&x| x == dim) {
                    d(nb + nlf + nrf + k)
                } else {
                    d(free_base + free.iter().position(|&x| x == dim).unwrap())
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let dims = (0..total).map(d).collect::<Vec<_>>().join(", ");
    let lhs_map = operand_map(lrank, &lb, &lc, &lfree, nb);
    let rhs_map = operand_map(rrank, &rb, &rc, &rfree, nb + nlf);
    let out_map = (0..nb + nlf + nrf).map(d).collect::<Vec<_>>().join(", ");
    let maps = format!(
        "[affine_map<({dims}) -> ({lhs_map})>, affine_map<({dims}) -> ({rhs_map})>, affine_map<({dims}) -> ({out_map})>]"
    );
    let iterators = (0..total)
        .map(|i| {
            if i < nb + nlf + nrf {
                "#linalg.iterator_type<parallel>"
            } else {
                "#linalg.iterator_type<reduction>"
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    let empty = rw.value(ods::tensor::empty(ctx, out_type, &[], loc).into());
    let zero = rw.value(arith::constant(ctx, zero_attribute(ctx, element).unwrap(), loc));
    let acc = rw.value(linalg_fill(ctx, zero, empty, loc));

    let body = Block::new(&[(element, loc), (element, loc), (element, loc)]);
    let (a, b, c) = (
        body.argument(0).unwrap().into(),
        body.argument(1).unwrap().into(),
        body.argument(2).unwrap().into(),
    );
    let m = body.value(build("arith.mulf", &[a, b], element, loc));
    let s = body.value(build("arith.addf", &[c, m], element, loc));
    body.append_operation(OperationBuilder::new("linalg.yield", loc).add_operands(&[s]).build().unwrap());
    let region = Region::new();
    region.append_block(body);

    let generic = rw.emit(
        OperationBuilder::new("linalg.generic", loc)
            .add_operands(&[lhs, rhs, acc])
            .add_results(&[out_type])
            .add_attributes(&[
                (Identifier::new(ctx, "indexing_maps"), Attribute::parse(ctx, &maps).unwrap()),
                (Identifier::new(ctx, "iterator_types"), Attribute::parse(ctx, &format!("[{iterators}]")).unwrap()),
                segments(ctx, &[2, 1]),
            ])
            .add_regions([region])
            .build()
            .unwrap(),
    );
    rw.replace_op_with_operation(op, generic);
    true
}

// impulse.for over tensor<i64> bounds -> scf.for over index. The body region is moved as-is;
// its induction variable is retyped to index and rebuilt as tensor<i64> for existing uses.
fn lower_for<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let index = Type::index(ctx);
    let bound = |i| {
        let v = scalar(ctx, rw, op.operand(i).unwrap(), loc);
        rw.value(op_cast("arith.index_cast", v, index, loc))
    };
    let (lb, ub, step) = (bound(0), bound(1), bound(2));
    let mut operands = vec![lb, ub, step];
    operands.extend(op.operands().skip(3));
    let result_types: Vec<Type> = op.results().map(|r| r.r#type()).collect();

    let region = Region::new();
    region.append_block(Block::new(&[]));
    let scf_for = rw.emit(
        OperationBuilder::new("scf.for", loc)
            .add_operands(&operands)
            .add_results(&result_types)
            .add_regions([region])
            .build()
            .unwrap(),
    );
    let body = move_region(rw, op, 0, scf_for, 0);
    let iv = body.argument(0).unwrap();
    iv.set_type(index);
    rw.set_insertion_point_to_start(body);
    let iv_i64 = rw.emit(op_cast("arith.index_cast", iv.into(), i64_type(ctx), loc));
    let iv_tensor = rw.value(
        ods::tensor::from_elements(
            ctx,
            RankedTensorType::new(&[], i64_type(ctx), None).into(),
            &[iv_i64.result(0).unwrap().into()],
            loc,
        )
        .into(),
    );
    rw.replace_all_uses_except(iv.into(), iv_tensor, iv_i64);

    replace_yield(ctx, rw, body, loc);
    rw.replace_op_with_operation(op, scf_for);
    true
}

fn lower_if<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let cond = scalar(ctx, rw, op.operand(0).unwrap(), loc);
    let result_types: Vec<Type> = op.results().map(|r| r.r#type()).collect();
    let regions = [Region::new(), Region::new()];
    for region in &regions {
        region.append_block(Block::new(&[]));
    }
    let scf_if = rw.emit(
        OperationBuilder::new("scf.if", loc)
            .add_operands(&[cond])
            .add_results(&result_types)
            .add_regions(regions)
            .build()
            .unwrap(),
    );
    for i in 0..2 {
        let body = move_region(rw, op, i, scf_if, i);
        replace_yield(ctx, rw, body, loc);
    }
    rw.replace_op_with_operation(op, scf_if);
    true
}

/// Moves region `from_index` of `from` into region `to_index` of `to`, whose single
/// placeholder block is discarded. Returns the moved entry block.
fn move_region<'c, 'a>(
    rw: &RewriterBase<'c, '_>,
    from: OperationRef<'c, '_>,
    from_index: usize,
    to: OperationRef<'c, 'a>,
    to_index: usize,
) -> melior::ir::BlockRef<'c, 'a> {
    let target = to.region(to_index).unwrap();
    let placeholder = target.first_block().unwrap();
    rw.inline_region_before(from.region(from_index).unwrap(), placeholder);
    rw.erase_block(placeholder);
    target.first_block().unwrap()
}

/// Replaces the block's `impulse.yield` terminator with `scf.yield`.
fn replace_yield<'c>(ctx: &'c Context, rw: &RewriterBase<'c, '_>, block: melior::ir::BlockRef<'c, '_>, loc: Location<'c>) {
    let terminator = block.terminator().unwrap();
    rw.set_insertion_point_before(terminator);
    let yielded: Vec<Value> = terminator.operands().collect();
    rw.emit(ods::scf::r#yield(ctx, &yielded, loc).into());
    rw.erase_op(terminator);
}

fn op_cast<'c>(name: &str, v: Value<'c, '_>, to: Type<'c>, loc: Location<'c>) -> Operation<'c> {
    build(name, &[v], to, loc)
}

// ── RNG ───────────────────────────────────────────────────────────────────────
//
// Counter-based: a tensor<2xui64> state (k0, k1) is hashed with the SplitMix64 finalizer.
// Streams are not bit-compatible with Enzyme-JAX, whose stablehlo.rng_bit_generator uses
// the backend's DEFAULT algorithm and makes no cross-backend guarantee either.

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix<'c: 's, 's, E: Emit<'c>>(ctx: &'c Context, e: &'s E, z: Value<'c, 's>, loc: Location<'c>) -> Value<'c, 's> {
    let t = i64_type(ctx);
    let step = |z: Value<'c, 's>, shift: i64, mul: Option<u64>| {
        let s = e.value(const_i64(ctx, shift, loc));
        let shifted = e.value(build("arith.shrui", &[z, s], t, loc));
        let x = e.value(build("arith.xori", &[z, shifted], t, loc));
        match mul {
            Some(m) => {
                let m = e.value(const_i64(ctx, m as i64, loc));
                e.value(build("arith.muli", &[x, m], t, loc))
            }
            None => x,
        }
    };
    let z = step(z, 30, Some(0xBF58_476D_1CE4_E5B9));
    let z = step(z, 27, Some(0x94D0_49BB_1331_11EB));
    step(z, 31, None)
}

fn add_const<'c: 's, 's, E: Emit<'c>>(ctx: &'c Context, e: &'s E, v: Value<'c, 's>, c: u64, loc: Location<'c>) -> Value<'c, 's> {
    let c = e.value(const_i64(ctx, c as i64, loc));
    e.value(build("arith.addi", &[v, c], i64_type(ctx), loc))
}

/// The two words of a tensor<2xui64> state as i64. `tensor.bitcast` has no bufferization, so the
/// unsigned words are extracted as scalars and cast; after LLVM conversion both are i64 and
/// `reconcile-unrealized-casts` removes the casts.
fn keys<'c, 's>(ctx: &'c Context, rw: &'s RewriterBase<'c, '_>, state: Value<'c, 's>, loc: Location<'c>) -> (Value<'c, 's>, Value<'c, 's>) {
    let word_type = element_type(state.r#type());
    let word = |i| {
        let i = rw.value(const_index(ctx, i, loc));
        let w = rw.value(ods::tensor::extract(ctx, word_type, state, &[i], loc).into());
        rw.value(cast(w, i64_type(ctx), loc))
    };
    (word(0), word(1))
}

fn cast<'c>(v: Value<'c, '_>, to: Type<'c>, loc: Location<'c>) -> Operation<'c> {
    build("builtin.unrealized_conversion_cast", &[v], to, loc)
}

/// The `n`th independent state derived from (k0, k1).
fn derive_state<'c, 's>(
    ctx: &'c Context,
    rw: &'s RewriterBase<'c, '_>,
    (k0, k1): (Value<'c, 's>, Value<'c, 's>),
    n: u64,
    state_type: Type<'c>,
    loc: Location<'c>,
) -> Value<'c, 's> {
    let t = i64_type(ctx);
    let word = |a: Value<'c, 's>, b: Value<'c, 's>, salt: u64| {
        let inner = mix(ctx, rw, add_const(ctx, rw, b, salt.wrapping_mul(GOLDEN), loc), loc);
        let sum = rw.value(build("arith.addi", &[a, inner], t, loc));
        mix(ctx, rw, sum, loc)
    };
    let word_type = element_type(state_type);
    let w0 = rw.value(cast(word(k0, k1, 2 * n + 1), word_type, loc));
    let w1 = rw.value(cast(word(k1, k0, 2 * n + 2), word_type, loc));
    rw.value(ods::tensor::from_elements(ctx, state_type, &[w0, w1], loc).into())
}

fn lower_random_split<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let state = op.operand(0).unwrap();
    let state_type = state.r#type();
    let k = keys(ctx, rw, state, loc);
    let outputs: Vec<Value> = (0..op.result_count() as u64)
        .map(|n| derive_state(ctx, rw, k, n, state_type, loc))
        .collect();
    rw.replace_op_with_values(op, &outputs);
    true
}

// impulse.random %state, %a, %b: UNIFORM samples [a, b); NORMAL samples mean a, stddev b.
fn lower_random<'c>(ctx: &'c Context, op: OperationRef<'c, '_>, rw: &RewriterBase<'c, '_>) -> bool {
    let loc = op.location();
    let distribution = op.attribute("rng_distribution").unwrap().to_string();
    let normal = distribution.contains("NORMAL") && !distribution.contains("MULTINORMAL");
    if !normal && !distribution.contains("UNIFORM") {
        return false;
    }
    let result_type = op.result(1).unwrap().r#type();
    let f64_type = Type::float64(ctx);
    if element_type(result_type) != f64_type {
        return false;
    }

    let state = op.operand(0).unwrap();
    let (k0, k1) = keys(ctx, rw, state, loc);
    let new_state = derive_state(ctx, rw, (k0, k1), 1 << 32, state.r#type(), loc);
    let a = scalar(ctx, rw, op.operand(1).unwrap(), loc);
    let b = scalar(ctx, rw, op.operand(2).unwrap(), loc);

    let dims = shape(result_type);
    let index = Type::index(ctx);
    let body = Block::new(&dims.iter().map(|_| (index, loc)).collect::<Vec<_>>());
    let t = i64_type(ctx);

    // Row-major linear index of this element, as the stream counter.
    let mut linear = body.value(const_index(ctx, 0, loc));
    for (i, &dim) in dims.iter().enumerate() {
        let dim = body.value(const_index(ctx, dim, loc));
        let scaled = body.value(build("arith.muli", &[linear, dim], index, loc));
        linear = body.value(build("arith.addi", &[scaled, body.argument(i).unwrap().into()], index, loc));
    }
    let counter = body.value(op_cast("arith.index_cast", linear, t, loc));
    let golden = body.value(const_i64(ctx, GOLDEN as i64, loc));
    let scaled = body.value(build("arith.muli", &[counter, golden], t, loc));
    let seeded = body.value(build("arith.addi", &[k0, scaled], t, loc));
    let mixed = mix(ctx, &body, seeded, loc);
    let keyed = body.value(build("arith.xori", &[mixed, k1], t, loc));
    let bits = mix(ctx, &body, keyed, loc);

    // Top 53 bits as a double in [0, 1).
    let unit = |bits: Value<'c, '_>| {
        let eleven = body.value(const_i64(ctx, 11, loc));
        let top = body.value(build("arith.shrui", &[bits, eleven], t, loc));
        let f = body.value(op_cast("arith.uitofp", top, f64_type, loc));
        let scale = body.value(const_f64(ctx, 1.0 / (1u64 << 53) as f64, loc));
        body.value(build("arith.mulf", &[f, scale], f64_type, loc))
    };
    let fop = |name, x, y| body.value(build(name, &[x, y], f64_type, loc));

    let sample = if normal {
        // Box–Muller with a second stream word.
        let golden = body.value(const_i64(ctx, GOLDEN as i64, loc));
        let bits2 = body.value(build("arith.xori", &[bits, golden], t, loc));
        let bits2 = mix(ctx, &body, bits2, loc);
        let (u1, u2) = (unit(bits), unit(bits2));
        let one = body.value(const_f64(ctx, 1.0, loc));
        let minus_two = body.value(const_f64(ctx, -2.0, loc));
        let two_pi = body.value(const_f64(ctx, std::f64::consts::TAU, loc));
        let one_minus = fop("arith.subf", one, u1);
        let log = body.value(op_cast("math.log", one_minus, f64_type, loc));
        let radius = body.value(op_cast("math.sqrt", fop("arith.mulf", minus_two, log), f64_type, loc));
        let angle = body.value(op_cast("math.cos", fop("arith.mulf", two_pi, u2), f64_type, loc));
        let z = fop("arith.mulf", radius, angle);
        fop("arith.addf", a, fop("arith.mulf", b, z))
    } else {
        let u = unit(bits);
        fop("arith.addf", a, fop("arith.mulf", fop("arith.subf", b, a), u))
    };
    body.append_operation(ods::tensor::r#yield(ctx, sample, loc).into());

    let region = Region::new();
    region.append_block(body);
    let values = rw.value(ods::tensor::generate(ctx, result_type, &[], region, loc).into());
    rw.replace_op_with_values(op, &[new_state, values]);
    true
}
