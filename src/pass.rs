//! Enzyme's passes, as melior `Pass`es.

use std::ffi::c_void;

use melior::{
    pass::{Pass, PassManager},
    Error, StringRef,
};
use mlir_sys::{mlirOpPassManagerAddPipeline, MlirStringRef};

use crate::{sys, utility::register_passes, Activity};

use crate::lower::{create_lower_broadcast_pass, create_lower_fill_zero_pass};

/// What `enzyme-wrap` generates, named as in Enzyme.
///
/// Enzyme also has split reverse mode, `ReverseModePrimal` and `ReverseModeGradient`, but its MLIR
/// side doesn't implement it: `ReverseModePrimal` reaches `llvm_unreachable`, which is undefined
/// behavior in a release build, so neither is offered here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeMode {
    /// The forward-mode derivative: tangents in, tangents out.
    ForwardMode,
    /// The reverse-mode derivative, forward and reverse passes in one function.
    ReverseModeCombined,
}

impl DerivativeMode {
    fn name(self) -> &'static str {
        match self {
            Self::ForwardMode => "ForwardMode",
            Self::ReverseModeCombined => "ReverseModeCombined",
        }
    }
}

/// `enzyme-wrap`: generates the derivative of the function `infn`, given the activity of each of
/// its arguments and results.
///
/// - `outfn: None` *replaces* `infn` with its derivative, under the same name, and makes it public:
///   the way to get a derivative to look up and call after JIT compilation.
/// - `outfn: Some(name)` adds the derivative alongside `infn` as `name`. Like the derivatives
///   Enzyme generates for `enzyme.autodiff`, it's private, meant to be called from other functions
///   in the module; cleanup passes like `symbol-dce` remove it if nothing calls it.
///
/// Enzyme's C API has no constructor for this pass, so this appends it to `manager` as a textual
/// pipeline element instead of returning a `Pass`. `enzyme-wrap` accepts only `Active`, `Dup`,
/// `Const` and `DupNoNeed`; the `NoNeed` variants of `Active` and `Const` are an error here.
///
/// Give one activity per argument and per result of `infn`. On a mismatch, Enzyme reports an error
/// diagnostic but doesn't fail the pass, so no derivative is generated and the run still succeeds.
pub fn add_differentiate_wrapper_pass(
    manager: &PassManager,
    infn: &str,
    outfn: Option<&str>,
    arg_activity: &[Activity],
    ret_activity: &[Activity],
    mode: DerivativeMode,
) -> Result<(), Error> {
    for name in std::iter::once(infn).chain(outfn) {
        if name.is_empty() || name.contains(|c: char| c.is_whitespace() || "{},=".contains(c)) {
            return Err(Error::ParsePassPipeline(format!("invalid function name {name:?}")));
        }
    }
    let element = format!(
        "enzyme-wrap{{infn={infn} outfn={} argTys={} retTys={} mode={}}}",
        outfn.unwrap_or(""),
        activity_list(arg_activity)?,
        activity_list(ret_activity)?,
        mode.name(),
    );
    add_pipeline_element(manager, &element)
}

// Enzyme's C API has no constructor for most of its passes, though it registers them all, so each of
// these appends its pass to a `PassManager` as a textual pipeline element, with the pass's default
// options, instead of returning a `Pass`.
macro_rules! pipeline_pass {
    ($(#[$doc:meta])* $name:ident, $element:literal) => {
        $(#[$doc])*
        ///
        /// Enzyme's C API has no constructor for this pass, so this appends it to `manager` as a
        /// textual pipeline element instead of returning a `Pass`.
        pub fn $name(manager: &PassManager) -> Result<(), Error> {
            add_pipeline_element(manager, $element)
        }
    };
}

pipeline_pass!(
    /// `enzyme-simplify-math`: simplifies the arithmetic Enzyme generates, such as additions of
    /// zero.
    add_mathematic_simplification_pass,
    "enzyme-simplify-math"
);
pipeline_pass!(
    /// `enzyme-activity-opt`: optimizes the activities of differentiated arguments and results,
    /// before differentiation.
    add_enzyme_activity_opt_pass,
    "enzyme-activity-opt"
);
pipeline_pass!(
    /// `flatten-enzyme-caches`: flattens multi-dimensional cache buffers to one dimension and
    /// lowers `enzyme.store` and `enzyme.load` to memref stores and loads.
    add_flatten_enzyme_caches_pass,
    "flatten-enzyme-caches"
);
pipeline_pass!(
    /// `simplify-memref-cache`: rewrites caches of memrefs of caches,
    /// `cache<memref<cache<...>>>`, to `cache<memref<...>>` where it can.
    add_simplify_memref_cache_pass,
    "simplify-memref-cache"
);
pipeline_pass!(
    /// `enzyme-batch-to-tensor`: lowers the batching ops `enzyme.concat` and `enzyme.extract`,
    /// which `enzyme-diff-batch` and some `width > 1` derivatives emit, to the Tensor dialect.
    add_enzyme_batch_to_tensor_pass,
    "enzyme-batch-to-tensor"
);
pipeline_pass!(
    /// `add-to-op-to-split`: removes `enzyme.addTo` from a reverse-mode `linalg.generic` with
    /// several outputs by duplicating the `linalg.generic` once per output.
    add_add_to_op_to_split_pass,
    "add-to-op-to-split"
);
pipeline_pass!(
    /// `split-multi-results`: splits region ops that yield several results into one op per result.
    add_split_multi_results_pass,
    "split-multi-results"
);
pipeline_pass!(
    /// `inline-enzyme-regions`: inlines `enzyme.autodiff` ops into `enzyme.autodiff_region` ops.
    add_inline_enzyme_into_region_pass,
    "inline-enzyme-regions"
);
pipeline_pass!(
    /// `outline-enzyme-regions`: outlines `enzyme.autodiff_region` ops into `enzyme.autodiff` ops.
    add_outline_enzyme_from_region_pass,
    "outline-enzyme-regions"
);
pipeline_pass!(
    /// `hoist-enzyme-regions`: hoists primal ops that don't depend on the differentiated values out
    /// of `enzyme.autodiff_region` ops.
    add_hoist_enzyme_from_region_pass,
    "hoist-enzyme-regions"
);
pipeline_pass!(
    /// `sink-checkpoint-views`: sinks pure views of mutable values into checkpointed loops.
    add_sink_checkpoint_views_pass,
    "sink-checkpoint-views"
);
pipeline_pass!(
    /// `hoist-loop-allocations`: hoists per-iteration scratch buffers out of sequential loops.
    add_hoist_loop_allocations_pass,
    "hoist-loop-allocations"
);
pipeline_pass!(
    /// `lower-affine-atomic-rmw`: lowers Enzyme's affine atomic read-modify-write ops to memref
    /// ones.
    add_lower_affine_atomic_rmw_pass,
    "lower-affine-atomic-rmw"
);
pipeline_pass!(
    /// `raise-llvm-ext`: raises LLVM dialect operations to Enzyme's LLVM Ext dialect.
    add_raise_llvm_ext_pass,
    "raise-llvm-ext"
);
pipeline_pass!(
    /// `lower-llvm-ext`: lowers Enzyme's LLVM Ext dialect to the LLVM dialect.
    add_lower_llvm_ext_pass,
    "lower-llvm-ext"
);

// Appends an Enzyme pass, written as a textual pipeline element, to `manager`.
fn add_pipeline_element(manager: &PassManager, element: &str) -> Result<(), Error> {
    register_passes();
    let mut message = String::new();
    let result = unsafe {
        mlirOpPassManagerAddPipeline(
            manager.as_operation_pass_manager().to_raw(),
            StringRef::new(element).to_raw(),
            Some(append_message),
            &mut message as *mut String as *mut c_void,
        )
    };
    if result.value == 0 {
        return Err(Error::ParsePassPipeline(message));
    }
    Ok(())
}

fn activity_list(activities: &[Activity]) -> Result<String, Error> {
    let names = activities
        .iter()
        .map(|activity| match activity {
            Activity::Active => Ok("enzyme_active"),
            Activity::Dup => Ok("enzyme_dup"),
            Activity::Const => Ok("enzyme_const"),
            Activity::DupNoNeed => Ok("enzyme_dupnoneed"),
            Activity::ActiveNoNeed | Activity::ConstNoNeed => Err(Error::ParsePassPipeline(
                format!("enzyme-wrap doesn't accept {activity:?}"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(names.join(","))
}

unsafe extern "C" fn append_message(text: MlirStringRef, data: *mut c_void) {
    let text = std::slice::from_raw_parts(text.data as *const u8, text.length);
    (*(data as *mut String)).push_str(&String::from_utf8_lossy(text));
}

/// `enzyme`: replaces `enzyme.autodiff`, `enzyme.fwddiff` and friends with calls to the
/// derivative functions it generates.
pub fn create_differentiate_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateDifferentiatePass) }
}

/// `enzyme`, running the textual pass pipeline `postpasses` on every derivative function it
/// generates, and verifying them afterwards if `verify_post_passes` is set.
pub fn create_differentiate_pass_with_options(postpasses: &str, verify_post_passes: bool) -> Pass {
    unsafe {
        Pass::from_raw(sys::enzymeCreateDifferentiatePassWithOptions(
            StringRef::new(postpasses).to_raw(),
            verify_post_passes,
        ))
    }
}

/// `convert-enzyme-to-memref`: lowers the Enzyme ops left after differentiation (gradients and
/// caches) to the MemRef dialect.
pub fn create_enzyme_ops_to_mem_ref_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateConvertEnzymeToMemRefPass) }
}

/// `remove-unnecessary-enzyme-ops`: where it can, replaces Enzyme's per-iteration caches and
/// gradient ops with plain values and buffers sized up front. Its buffers are accessed with
/// `enzyme.store` and `enzyme.load`, which [`add_flatten_enzyme_caches_pass`] lowers.
pub fn create_remove_unused_enzyme_ops_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateRemoveUnusedEnzymeOpsPass) }
}

/// Adds the passes that replace every Enzyme op in a module with upstream MLIR:
///
/// 1. Generates the derivatives: replaces `enzyme.batch`, `enzyme.autodiff` and `enzyme.fwddiff`
///    with calls to the functions Enzyme generates, which are left as private functions.
/// 2. Optimizes them: where it can, replaces Enzyme's per-iteration loop caches with buffers sized
///    up front, then lowers those buffers' accesses to memref loads and stores.
/// 3. Lowers what's left: `enzyme.concat` and `enzyme.extract` to tensors, `enzyme.fill_zero` and
///    `enzyme.broadcast` with this crate's stopgap passes, and Enzyme's gradient and cache ops to
///    memrefs.
///
/// Afterwards the module uses only upstream dialects, including `linalg`.
///
/// For `enzyme-wrap`, add [`add_differentiate_wrapper_pass`] to `manager` first.
pub fn add_enzyme_lowering(manager: &PassManager) -> Result<(), Error> {
    // 1.
    manager.add_pass(create_batch_pass());
    manager.add_pass(create_differentiate_pass());
    // 2. `remove-unnecessary-enzyme-ops` writes the buffers with `enzyme.store` and reads them with
    //    `enzyme.load`, which only `flatten-enzyme-caches` lowers.
    manager.add_pass(create_remove_unused_enzyme_ops_pass());
    add_flatten_enzyme_caches_pass(manager)?;
    // 3. Enzyme has no lowering for `enzyme.fill_zero` or `enzyme.broadcast` (EnzymeAD/Enzyme#3884),
    //    and `convert-enzyme-to-memref` fails on any Enzyme op it can't convert, so this crate's
    //    passes lower those two first.
    add_enzyme_batch_to_tensor_pass(manager)?;
    manager.add_pass(create_lower_fill_zero_pass());
    manager.add_pass(create_lower_broadcast_pass());
    manager.add_pass(create_enzyme_ops_to_mem_ref_pass());
    Ok(())
}

/// `enzyme-batch`: replaces `enzyme.batch` with a call to a batched copy of the function.
pub fn create_batch_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateBatchPass) }
}

/// `enzyme-diff-batch`: merges `enzyme.fwddiff` and `enzyme.autodiff` calls on the same function
/// into one batched call.
pub fn create_batch_diff_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateBatchDiffPass) }
}
