//! Draft wrappers for the Impulse dialect C API.
//!
//! These are parked here because the Impulse dialect (`impulse.*`) is a
//! separate dialect from Enzyme core (`enzyme.*`) and requires its own dialect
//! registration that isn't yet wired into our `setup_context`. Revisit once
//! we add an `impulse_dialect_handle()` to the C API and link MLIRImpulse.

use mlir_sys::MlirAttribute;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RngDistribution {
    Uniform = 0,
    Normal = 1,
    MultiNormal = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportKind {
    Real = 0,
    Positive = 1,
    UnitInterval = 2,
    Interval = 3,
    GreaterThan = 4,
    LessThan = 5,
}

pub unsafe fn rng_distribution_attr(
    ctx: mlir_sys::MlirContext,
    distribution: RngDistribution,
) -> MlirAttribute {
    todo!("wire up enzymeRngDistributionAttrGet once Impulse dialect is loaded")
}

pub unsafe fn support_attr(
    ctx: mlir_sys::MlirContext,
    kind: SupportKind,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> MlirAttribute {
    todo!("wire up enzymeSupportAttrGet once Impulse dialect is loaded")
}

/// Construct an `impulse.hmc_config` attribute.
///
/// Attaches to `impulse.SimulateOp` to select HMC as the MCMC algorithm.
/// `trajectory_length` sets the leapfrog integration horizon; the two booleans
/// enable step-size and mass-matrix adaptation during warmup.
pub unsafe fn hmc_config_attr(
    ctx: mlir_sys::MlirContext,
    trajectory_length: f64,
    adapt_step_size: bool,
    adapt_mass_matrix: bool,
) -> MlirAttribute {
    todo!("wire up enzymeHMCConfigAttrGet once Impulse dialect is loaded")
}

/// Construct an `impulse.nuts_config` attribute.
///
/// Attaches to `impulse.SimulateOp` to select NUTS as the MCMC algorithm.
/// `max_delta_energy`, when `Some`, caps the energy deviation used to prune the
/// binary tree; `None` disables the cap.
pub unsafe fn nuts_config_attr(
    ctx: mlir_sys::MlirContext,
    max_tree_depth: i64,
    max_delta_energy: Option<f64>,
    adapt_step_size: bool,
    adapt_mass_matrix: bool,
) -> MlirAttribute {
    todo!("wire up enzymeNUTSConfigAttrGet once Impulse dialect is loaded")
}

/// Construct an `impulse.symbol` attribute from a raw function pointer.
///
/// Used by `impulse.SampleOp` to identify the log-density function at the MLIR
/// attribute level. `ptr` must remain valid for the lifetime of the MLIR context.
pub unsafe fn symbol_attr(ctx: mlir_sys::MlirContext, ptr: u64) -> MlirAttribute {
    todo!("wire up enzymeSymbolAttrGet once Impulse dialect is loaded")
}
