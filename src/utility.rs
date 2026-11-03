//! Registration and setup, like `melior::utility`.

use melior::{dialect::DialectRegistry, utility::register_all_dialects, Context};

use crate::{dialect_handle, sys};

/// Registers Enzyme's AD interface models for upstream dialects (arith, func, scf, ...).
/// Must run before the registry is used to create a context.
pub fn register_dialect_extensions(registry: &DialectRegistry) {
    unsafe { sys::enzymeRegisterDialectExtensions(registry.to_raw()) }
}

/// Registers Enzyme's passes with MLIR's global registry, so a textual pass
/// pipeline can name them (`enzyme`, `add-to-op-to-split`, ...). Safe to call
/// more than once.
pub fn register_passes() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe { sys::enzymeRegisterPasses() });
}

/// A context with all upstream MLIR dialects and the Enzyme dialect loaded.
pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    dialect_handle().insert_dialect(&registry);
    register_dialect_extensions(&registry);
    let context = Context::new_with_registry(&registry, false);
    context.load_all_available_dialects();
    context
}
