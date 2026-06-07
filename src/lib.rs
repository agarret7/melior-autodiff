//! Melior-facing support for Enzyme's MLIR C API.
//!
//! This is not an extension of Enzyme itself. It is a small Rust/Melior
//! integration layer: bindgen still generates the raw ABI declarations into
//! Cargo's `OUT_DIR`, while this module exposes the intentional surface callers
//! should use from Rust.

mod sys {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code,
        unused_imports
    )]

    include!(concat!(env!("OUT_DIR"), "/enzyme_bindings.rs"));
}

pub use sys::{
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass,
    enzymeCreateDifferentiatePassWithOptions, enzymeRegisterDialectExtensions,
    enzymeRegisterPasses,
};

fn mlir_string_ref(s: &str) -> mlir_sys::MlirStringRef {
    mlir_sys::MlirStringRef {
        data: s.as_ptr() as *const std::os::raw::c_char,
        length: s.len(),
    }
}

pub unsafe fn enzyme_dialect_handle() -> mlir_sys::MlirDialectHandle {
    sys::mlirGetDialectHandle__enzyme__()
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activity {
    Active = 0,
    Dup = 1,
    Const = 2,
    DupNoNeed = 3,
    ActiveNoNeed = 4,
    ConstNoNeed = 5,
}

pub unsafe fn activity_attr(
    ctx: mlir_sys::MlirContext,
    activity: Activity,
) -> mlir_sys::MlirAttribute {
    unsafe { sys::enzymeActivityAttrGet(ctx, activity as u32) }
}

/// Look up a JIT-compiled function and return it as a `Box<dyn Fn>`.
///
/// The transmute is contained inside the macro; the returned closure is safe to call.
/// The signature must match the compiled function exactly or the call is UB.
///
/// # Example
/// ```ignore
/// let grad = lookup_jit_fn!(&engine, "dsquare", fn(x: f64, dr: f64) -> f64);
/// let result = grad(3.0, 1.0); // 6.0
/// ```
#[macro_export]
macro_rules! lookup_jit_fn {
    ($engine:expr, $name:expr, fn($($a:ident : $A:ty),*) -> $R:ty) => {{
        let f: unsafe extern "C" fn($($A),*) -> $R = unsafe {
            let raw = $engine.lookup($name);
            assert!(!raw.is_null(), "lookup_jit_fn: {} returned null", $name);
            std::mem::transmute(raw)
        };
        Box::new(move |$($a: $A),*| unsafe { f($($a),*) }) as Box<dyn Fn($($A),*) -> $R>
    }};
}

pub unsafe fn create_autodiff_op(
    ctx: mlir_sys::MlirContext,
    function: &str,
    result_types: &[mlir_sys::MlirType],
    inputs: &[mlir_sys::MlirValue],
    activity: &[mlir_sys::MlirAttribute],
    ret_activity: &[mlir_sys::MlirAttribute],
    width: i64,
    strong_zero: bool,
    loc: mlir_sys::MlirLocation,
) -> mlir_sys::MlirOperation {
    unsafe {
        sys::enzymeAutoDiffOpCreate(
            ctx,
            mlir_string_ref(function),
            result_types.as_ptr(),
            result_types.len() as isize,
            inputs.as_ptr(),
            inputs.len() as isize,
            activity.as_ptr(),
            activity.len() as isize,
            ret_activity.as_ptr(),
            ret_activity.len() as isize,
            width,
            strong_zero,
            loc,
        )
    }
}

pub unsafe fn create_fwddiff_op(
    ctx: mlir_sys::MlirContext,
    function: &str,
    result_types: &[mlir_sys::MlirType],
    inputs: &[mlir_sys::MlirValue],
    activity: &[mlir_sys::MlirAttribute],
    ret_activity: &[mlir_sys::MlirAttribute],
    width: i64,
    strong_zero: bool,
    loc: mlir_sys::MlirLocation,
) -> mlir_sys::MlirOperation {
    unsafe {
        sys::enzymeForwardDiffOpCreate(
            ctx,
            mlir_string_ref(function),
            result_types.as_ptr(),
            result_types.len() as isize,
            inputs.as_ptr(),
            inputs.len() as isize,
            activity.as_ptr(),
            activity.len() as isize,
            ret_activity.as_ptr(),
            ret_activity.len() as isize,
            width,
            strong_zero,
            loc,
        )
    }
}

pub unsafe fn create_jacobian_op(
    ctx: mlir_sys::MlirContext,
    function: &str,
    result_types: &[mlir_sys::MlirType],
    inputs: &[mlir_sys::MlirValue],
    activity: &[mlir_sys::MlirAttribute],
    ret_activity: &[mlir_sys::MlirAttribute],
    width: i64,
    strong_zero: bool,
    loc: mlir_sys::MlirLocation,
) -> mlir_sys::MlirOperation {
    unsafe {
        sys::enzymeJacobianOpCreate(
            ctx,
            mlir_string_ref(function),
            result_types.as_ptr(),
            result_types.len() as isize,
            inputs.as_ptr(),
            inputs.len() as isize,
            activity.as_ptr(),
            activity.len() as isize,
            ret_activity.as_ptr(),
            ret_activity.len() as isize,
            width,
            strong_zero,
            loc,
        )
    }
}
