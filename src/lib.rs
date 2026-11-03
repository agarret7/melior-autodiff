//! Melior-facing support for Enzyme's MLIR C API.
//!
//! This is not an extension of Enzyme itself. It is a small Rust/Melior
//! integration layer: bindgen still generates the raw ABI declarations into
//! Cargo's `OUT_DIR`, while this module exposes the intentional surface callers
//! should use from Rust.

use melior::{
    dialect::DialectHandle,
    ir::{
        attribute::{ArrayAttribute, BoolAttribute, FlatSymbolRefAttribute, IntegerAttribute},
        r#type::IntegerType,
        Attribute, Location, Operation, Type, Value,
    },
    Context,
};

// Runs the README's examples as doctests, so they keep compiling.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod lower;
pub mod ods;
pub mod pass;
pub mod utility;

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

/// The Enzyme dialect, to insert into a `DialectRegistry` or load into a `Context`.
pub fn dialect_handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(sys::mlirGetDialectHandle__enzyme__()) }
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

// `autodiff`, `fwddiff` and `jacobian` take the same arguments; each wraps its generated builder,
// naming the function and the activities directly. `width` and `strong_zero` are left unset at
// their defaults (1 and false), as the generated op would leave them.
macro_rules! differentiation_op {
    ($(#[$doc:meta])* $name:ident, $op:ident) => {
        $(#[$doc])*
        #[allow(clippy::too_many_arguments)]
        pub fn $name<'c>(
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
            let mut builder = ods::$op::builder(context, location)
                .outputs(result_types)
                .inputs(operands)
                .r#fn(FlatSymbolRefAttribute::new(context, function))
                .activity(activity_array_attribute(context, activity))
                .ret_activity(activity_array_attribute(context, ret_activity));
            if width != 1 {
                let i64_type = IntegerType::new(context, 64).into();
                builder = builder.width(IntegerAttribute::new(i64_type, width));
            }
            if strong_zero {
                builder = builder.strong_zero(BoolAttribute::new(context, true));
            }
            builder.build().into()
        }
    };
}

differentiation_op!(
    /// Builds `enzyme.autodiff` (reverse mode) on the function named `function`.
    autodiff,
    AutoDiffOperation
);
differentiation_op!(
    /// Builds `enzyme.fwddiff` (forward mode) on the function named `function`.
    fwddiff,
    ForwardDiffOperation
);
differentiation_op!(
    /// Builds `enzyme.jacobian` on the function named `function`.
    jacobian,
    JacobianOperation
);
