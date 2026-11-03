//! Enzyme's passes, as melior `Pass`es.

use melior::{pass::Pass, StringRef};

use crate::sys;

pub use crate::lower::create_lower_enzyme_helpers_pass;

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
pub fn create_convert_enzyme_to_memref_pass() -> Pass {
    unsafe { Pass::from_raw_fn(sys::enzymeCreateConvertEnzymeToMemRefPass) }
}

/// `remove-unnecessary-enzyme-ops`: removes Enzyme ops that differentiation left unnecessary.
pub fn create_remove_unnecessary_enzyme_ops_pass() -> Pass {
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
