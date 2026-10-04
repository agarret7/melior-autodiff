//! Melior-facing support for Enzyme's MLIR C API.
//!
//! This is not an extension of Enzyme itself. It is a small Rust/Melior
//! integration layer: bindgen still generates the raw ABI declarations into
//! Cargo's `OUT_DIR`, while this module exposes the intentional surface callers
//! should use from Rust.

use melior::{
    dialect::DialectRegistry,
    ir::{
        attribute::{ArrayAttribute, AttributeLike},
        Attribute, Location, Operation, Type, TypeLike, Value, ValueLike,
    },
    utility::register_all_dialects,
    Context, StringRef,
};
use mlir_sys::{
    MlirAttribute, MlirContext, MlirLocation, MlirOperation, MlirStringRef, MlirType, MlirValue,
};

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
    enzymeCreateBatchDiffPass, enzymeCreateBatchPass, enzymeCreateConvertEnzymeToMemRefPass,
    enzymeCreateDifferentiatePass, enzymeCreateDifferentiatePassWithOptions,
    enzymeCreateRemoveUnusedEnzymeOpsPass,
};

/// Registers Enzyme's AD interface models for upstream dialects (arith, func, scf, ...).
/// Must run before the registry is used to create a context.
pub fn register_dialect_extensions(registry: &DialectRegistry) {
    unsafe { sys::enzymeRegisterDialectExtensions(registry.to_raw()) }
}

pub fn load_dialect(context: &Context) {
    unsafe {
        mlir_sys::mlirDialectHandleLoadDialect(
            sys::mlirGetDialectHandle__enzyme__(),
            context.to_raw(),
        );
    }
}

/// A context with all upstream MLIR dialects and the Enzyme dialect loaded.
pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    register_dialect_extensions(&registry);
    let context = Context::new_with_registry(&registry, false);
    context.load_all_available_dialects();
    load_dialect(&context);
    context
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

pub fn activity_attribute(context: &Context, activity: Activity) -> Attribute<'_> {
    unsafe {
        Attribute::from_raw(sys::enzymeActivityAttrGet(
            context.to_raw(),
            activity as u32,
        ))
    }
}

pub fn activity_array_attribute<'c>(
    context: &'c Context,
    activities: &[Activity],
) -> ArrayAttribute<'c> {
    let attributes = activities
        .iter()
        .map(|activity| activity_attribute(context, *activity))
        .collect::<Vec<_>>();
    ArrayAttribute::new(context, &attributes)
}

/// Builds `enzyme.autodiff` (reverse mode) on the function named `function`.
pub fn autodiff<'c>(
    context: &'c Context,
    function: &str,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: &[Activity],
    ret_activity: &[Activity],
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_op(
        sys::enzymeAutoDiffOpCreate,
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Builds `enzyme.fwddiff` (forward mode) on the function named `function`.
pub fn fwddiff<'c>(
    context: &'c Context,
    function: &str,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: &[Activity],
    ret_activity: &[Activity],
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_op(
        sys::enzymeForwardDiffOpCreate,
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Builds `enzyme.jacobian` on the function named `function`.
pub fn jacobian<'c>(
    context: &'c Context,
    function: &str,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: &[Activity],
    ret_activity: &[Activity],
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_op(
        sys::enzymeJacobianOpCreate,
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Builds `enzyme.batch`, mapping the function named `function` over leading `batch_shape` dims.
pub fn batch<'c>(
    context: &'c Context,
    function: &str,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    batch_shape: &[i64],
    location: Location<'c>,
) -> Operation<'c> {
    let operands = raw_values(operands);
    let result_types = raw_types(result_types);
    unsafe {
        Operation::from_raw(sys::enzymeBatchOpCreate(
            context.to_raw(),
            StringRef::new(function).to_raw(),
            result_types.as_ptr(),
            result_types.len() as isize,
            operands.as_ptr(),
            operands.len() as isize,
            batch_shape.as_ptr(),
            batch_shape.len() as isize,
            location.to_raw(),
        ))
    }
}

type DiffOpCreate = unsafe extern "C" fn(
    MlirContext,
    MlirStringRef,
    *const MlirType,
    isize,
    *const MlirValue,
    isize,
    *const MlirAttribute,
    isize,
    *const MlirAttribute,
    isize,
    i64,
    bool,
    MlirLocation,
) -> MlirOperation;

fn diff_op<'c>(
    create: DiffOpCreate,
    context: &'c Context,
    function: &str,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: &[Activity],
    ret_activity: &[Activity],
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    let operands = raw_values(operands);
    let result_types = raw_types(result_types);
    let activity = raw_activities(context, activity);
    let ret_activity = raw_activities(context, ret_activity);
    unsafe {
        Operation::from_raw(create(
            context.to_raw(),
            StringRef::new(function).to_raw(),
            result_types.as_ptr(),
            result_types.len() as isize,
            operands.as_ptr(),
            operands.len() as isize,
            activity.as_ptr(),
            activity.len() as isize,
            ret_activity.as_ptr(),
            ret_activity.len() as isize,
            width,
            strong_zero,
            location.to_raw(),
        ))
    }
}

fn raw_values(values: &[Value]) -> Vec<MlirValue> {
    values.iter().map(|value| value.to_raw()).collect()
}

fn raw_types(types: &[Type]) -> Vec<MlirType> {
    types.iter().map(|ty| ty.to_raw()).collect()
}

fn raw_activities(context: &Context, activities: &[Activity]) -> Vec<MlirAttribute> {
    activities
        .iter()
        .map(|activity| activity_attribute(context, *activity).to_raw())
        .collect()
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
