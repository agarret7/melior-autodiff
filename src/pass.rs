//! Enzyme's passes, as melior `Pass`es.

use std::ffi::c_void;

use melior::{
    pass::{Pass, PassManager},
    Error, StringRef,
};
use mlir_sys::{mlirOpPassManagerAddPipeline, MlirStringRef};

use crate::{sys, utility::register_passes, Activity};

pub use crate::lower::create_lower_enzyme_helpers_pass;

/// What `enzyme-wrap` generates, named as in Enzyme.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeMode {
    /// The forward-mode derivative: tangents in, tangents out.
    ForwardMode,
    /// The reverse-mode derivative, forward and reverse passes in one function.
    ReverseModeCombined,
    /// Only the forward pass of reverse mode, which records what the reverse pass needs.
    ReverseModePrimal,
    /// Only the reverse pass of reverse mode.
    ReverseModeGradient,
}

impl DerivativeMode {
    fn name(self) -> &'static str {
        match self {
            Self::ForwardMode => "ForwardMode",
            Self::ReverseModeCombined => "ReverseModeCombined",
            Self::ReverseModePrimal => "ReverseModePrimal",
            Self::ReverseModeGradient => "ReverseModeGradient",
        }
    }
}

/// `enzyme-wrap`: generates `outfn`, the derivative of the function `infn`, given the activity of
/// each of its arguments and results. `outfn` is private.
///
/// Enzyme's C API has no constructor for this pass, so this appends it to `manager` as a textual
/// pipeline element instead of returning a `Pass`. `enzyme-wrap` accepts only `Active`, `Dup`,
/// `Const` and `DupNoNeed`; the `NoNeed` variants of `Active` and `Const` are an error here.
///
/// Give one activity per argument and per result of `infn`. On a mismatch, Enzyme reports an error
/// diagnostic but doesn't fail the pass, so no `outfn` is generated and the run still succeeds.
pub fn add_differentiate_wrapper_pass(
    manager: &PassManager,
    infn: &str,
    outfn: &str,
    arg_activity: &[Activity],
    ret_activity: &[Activity],
    mode: DerivativeMode,
) -> Result<(), Error> {
    for name in [infn, outfn] {
        if name.is_empty() || name.contains(|c: char| c.is_whitespace() || "{},=".contains(c)) {
            return Err(Error::ParsePassPipeline(format!("invalid function name {name:?}")));
        }
    }
    let element = format!(
        "enzyme-wrap{{infn={infn} outfn={outfn} argTys={} retTys={} mode={}}}",
        activity_list(arg_activity)?,
        activity_list(ret_activity)?,
        mode.name(),
    );
    add_pipeline_element(manager, &element)
}

/// `enzyme-simplify-math`: simplifies the arithmetic Enzyme generates, such as additions of zero.
///
/// Enzyme's C API has no constructor for this pass, so this appends it to `manager` as a textual
/// pipeline element instead of returning a `Pass`.
pub fn add_mathematic_simplification_pass(manager: &PassManager) -> Result<(), Error> {
    add_pipeline_element(manager, "enzyme-simplify-math")
}

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

/// `remove-unnecessary-enzyme-ops`: removes Enzyme ops that differentiation left unnecessary.
pub fn create_remove_unused_enzyme_ops_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateRemoveUnusedEnzymeOpsPass) }
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
