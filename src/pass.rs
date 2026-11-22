//! Enzyme's passes, as melior `Pass`es.

use std::ffi::c_void;

use melior::{
    Error, StringRef,
    pass::{Pass, PassManager},
};
use mlir_sys::{MlirStringRef, mlirOpPassManagerAddPipeline};

use crate::{Activity, sys, utility::register_passes};

use crate::lower::create_lower_broadcast_pass;

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
/// Enzyme's C API can't set a pass's options, so this appends `enzyme-wrap` to `manager` as a
/// textual pipeline element instead of returning a `Pass`. `enzyme-wrap` accepts only `Active`, `Dup`,
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
            return Err(Error::ParsePassPipeline(format!(
                "invalid function name {name:?}"
            )));
        }
    }
    let element = format!(
        "enzyme-wrap{{infn={infn} outfn={} argTys={} retTys={} mode={}}}",
        outfn.unwrap_or(""),
        activity_list(arg_activity)?,
        activity_list(ret_activity)?,
        mode.name(),
    );
    // A textual element names a pass through the registry.
    register_passes();
    let mut message = String::new();
    let result = unsafe {
        mlirOpPassManagerAddPipeline(
            manager.as_operation_pass_manager().to_raw(),
            StringRef::new(&element).to_raw(),
            Some(append_message),
            &mut message as *mut String as *mut c_void,
        )
    };
    if result.value == 0 {
        return Err(Error::ParsePassPipeline(message));
    }
    Ok(())
}

// Each wraps the constructor Enzyme's C API generates for a pass, which creates it with its
// default options.
macro_rules! enzyme_pass {
    ($(#[$doc:meta])* $name:ident, $constructor:ident) => {
        $(#[$doc])*
        pub fn $name() -> Pass {
            unsafe { Pass::from_raw_fn(sys::$constructor) }
        }
    };
}

enzyme_pass!(
    /// `enzyme`: replaces `enzyme.autodiff`, `enzyme.fwddiff` and friends with calls to the
    /// derivative functions it generates.
    create_differentiate_pass,
    mlirCreateEnzymeDifferentiatePass
);
enzyme_pass!(
    /// `enzyme-batch`: replaces `enzyme.batch` with a call to a batched copy of the function.
    create_batch_pass,
    mlirCreateEnzymeBatchPass
);
enzyme_pass!(
    /// `enzyme-diff-batch`: merges `enzyme.fwddiff` and `enzyme.autodiff` calls on the same
    /// function into one batched call.
    create_batch_diff_pass,
    mlirCreateEnzymeBatchDiffPass
);
enzyme_pass!(
    /// `enzyme-activity-opt`: optimizes the activities of differentiated arguments and results,
    /// before differentiation.
    create_enzyme_activity_opt_pass,
    mlirCreateEnzymeEnzymeActivityOpt
);
enzyme_pass!(
    /// `enzyme-simplify-math`: simplifies the arithmetic Enzyme generates, such as additions of
    /// zero.
    create_mathematic_simplification_pass,
    mlirCreateEnzymeMathematicSimplificationPass
);
enzyme_pass!(
    /// `remove-unnecessary-enzyme-ops`: where it can, replaces Enzyme's per-iteration caches and
    /// gradient ops with plain values and buffers sized up front. Its buffers are accessed with
    /// `enzyme.store` and `enzyme.load`, which [`create_flatten_enzyme_caches_pass`] lowers.
    create_remove_unused_enzyme_ops_pass,
    mlirCreateEnzymeRemoveUnusedEnzymeOpsPass
);
enzyme_pass!(
    /// `flatten-enzyme-caches`: flattens multi-dimensional cache buffers to one dimension and
    /// lowers `enzyme.store` and `enzyme.load` to memref stores and loads.
    create_flatten_enzyme_caches_pass,
    mlirCreateEnzymeFlattenEnzymeCachesPass
);
enzyme_pass!(
    /// `simplify-memref-cache`: rewrites caches of memrefs of caches,
    /// `cache<memref<cache<...>>>`, to `cache<memref<...>>` where it can.
    create_simplify_memref_cache_pass,
    mlirCreateEnzymeSimplifyMemrefCachePass
);
enzyme_pass!(
    /// `enzyme-batch-to-tensor`: lowers the batching ops `enzyme.concat` and `enzyme.extract`,
    /// which `enzyme-diff-batch` and some `width > 1` derivatives emit, to the Tensor dialect.
    create_enzyme_batch_to_tensor_pass,
    mlirCreateEnzymeEnzymeBatchToTensorPass
);
enzyme_pass!(
    /// `convert-enzyme-to-memref`: lowers the Enzyme ops left after differentiation (gradients and
    /// caches) to the MemRef dialect.
    create_enzyme_ops_to_mem_ref_pass,
    mlirCreateEnzymeEnzymeOpsToMemRefPass
);
enzyme_pass!(
    /// `add-to-op-to-split`: removes `enzyme.addTo` from a reverse-mode `linalg.generic` with
    /// several outputs by duplicating the `linalg.generic` once per output.
    create_add_to_op_to_split_pass,
    mlirCreateEnzymeAddToOpToSplitPass
);
enzyme_pass!(
    /// `split-multi-results`: splits region ops that yield several results into one op per result.
    create_split_multi_results_pass,
    mlirCreateEnzymeSplitMultiResultsPass
);
enzyme_pass!(
    /// `inline-enzyme-regions`: inlines `enzyme.autodiff` ops into `enzyme.autodiff_region` ops.
    create_inline_enzyme_into_region_pass,
    mlirCreateEnzymeInlineEnzymeIntoRegionPass
);
enzyme_pass!(
    /// `outline-enzyme-regions`: outlines `enzyme.autodiff_region` ops into `enzyme.autodiff` ops.
    create_outline_enzyme_from_region_pass,
    mlirCreateEnzymeOutlineEnzymeFromRegionPass
);
enzyme_pass!(
    /// `hoist-enzyme-regions`: hoists primal ops that don't depend on the differentiated values out
    /// of `enzyme.autodiff_region` ops.
    create_hoist_enzyme_from_region_pass,
    mlirCreateEnzymeHoistEnzymeFromRegionPass
);
enzyme_pass!(
    /// `sink-checkpoint-views`: sinks pure views of mutable values into checkpointed loops.
    create_sink_checkpoint_views_pass,
    mlirCreateEnzymeSinkCheckpointViewsPass
);
enzyme_pass!(
    /// `hoist-loop-allocations`: hoists per-iteration scratch buffers out of sequential loops.
    create_hoist_loop_allocations_pass,
    mlirCreateEnzymeHoistLoopAllocationsPass
);
enzyme_pass!(
    /// `lower-affine-atomic-rmw`: replaces `enzyme.affine_atomic_rmw` with `enzyme.atomic_rmw`,
    /// which indexes its memref directly. Nothing in Enzyme lowers `enzyme.atomic_rmw` further.
    create_lower_affine_atomic_rmw_pass,
    mlirCreateEnzymeLowerAffineAtomicRmwPass
);
enzyme_pass!(
    /// `lower-enzyme-binomial-progress`: lowers `enzyme.binomial_progress`, which Enzyme emits for
    /// loops with binomial checkpointing, to `scf` and `arith`.
    create_lower_binomial_progress_pass,
    mlirCreateEnzymeLowerBinomialProgressPass
);
enzyme_pass!(
    /// `raise-llvm-ext`: raises LLVM dialect operations to Enzyme's LLVM Ext dialect.
    create_raise_llvm_ext_pass,
    mlirCreateEnzymeRaiseLLVMExtPass
);
enzyme_pass!(
    /// `lower-llvm-ext`: lowers Enzyme's LLVM Ext dialect to the LLVM dialect.
    create_lower_llvm_ext_pass,
    mlirCreateEnzymeLowerLLVMExtPass
);

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
    // SAFETY: MLIR passes a valid string, and `data` is the `String` that
    // `add_differentiate_wrapper_pass` hands it.
    unsafe {
        let text = std::slice::from_raw_parts(text.data as *const u8, text.length);
        (*(data as *mut String)).push_str(&String::from_utf8_lossy(text));
    }
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

/// Adds the passes that lower Enzyme's ops in a module to upstream MLIR, including `linalg`: they
/// generate the derivatives, replace Enzyme's loop caches with buffers sized up front where they
/// can, and lower the remaining Enzyme ops.
///
/// Not every module with valid Enzyme ops lowers. `enzyme.jacobian` is unsupported, and so is
/// reverse mode through parallel loops (`scf.parallel`, `affine.parallel`), whose derivatives
/// accumulate with `enzyme.atomic_rmw`, which Enzyme has no lowering for. A module that needs
/// either fails to lower with these passes.
///
/// The passes this adds can change between releases of this crate or of Enzyme. For a pipeline
/// that stays fixed, add the passes yourself with the `create_*_pass` functions.
///
/// For `enzyme-wrap`, add [`add_differentiate_wrapper_pass`] to `manager` first.
pub fn add_enzyme_lowering(manager: &PassManager) {
    // Outlining turns the region ops into `enzyme.autodiff` and `enzyme.fwddiff`, which the passes
    // after it differentiate.
    manager.add_pass(create_outline_enzyme_from_region_pass());
    manager.add_pass(create_batch_pass());
    manager.add_pass(create_differentiate_pass());
    // `remove-unnecessary-enzyme-ops` writes the buffers with `enzyme.store` and reads them with
    // `enzyme.load`, which only `flatten-enzyme-caches` lowers.
    manager.add_pass(create_remove_unused_enzyme_ops_pass());
    manager.add_pass(create_flatten_enzyme_caches_pass());
    manager.add_pass(create_lower_binomial_progress_pass());
    manager.add_pass(create_enzyme_batch_to_tensor_pass());
    // Enzyme has no lowering for `enzyme.broadcast` (EnzymeAD/Enzyme#3884), and
    // `convert-enzyme-to-memref` fails on any Enzyme op it can't convert, so this crate's pass
    // lowers it first.
    manager.add_pass(create_lower_broadcast_pass());
    manager.add_pass(create_enzyme_ops_to_mem_ref_pass());
}
