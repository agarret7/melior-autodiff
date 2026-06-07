use melior::ir::{BlockLike, Location, Module};
use melior_autodiff::{activity_attr, create_autodiff_op, Activity};
use mlir_sys::{mlirF64TypeGet, mlirLocationUnknownGet};

mod common;
use common::setup_context;

#[test]
fn activity_attrs_cover_all_variants() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let autodiff_op = unsafe {
        let raw_ctx = ctx.to_raw();
        let raw_loc = mlirLocationUnknownGet(raw_ctx);
        let result_types = [mlirF64TypeGet(raw_ctx)];
        let activity_arr = [
            activity_attr(raw_ctx, Activity::Active),
            activity_attr(raw_ctx, Activity::Dup),
            activity_attr(raw_ctx, Activity::Const),
            activity_attr(raw_ctx, Activity::DupNoNeed),
            activity_attr(raw_ctx, Activity::ActiveNoNeed),
            activity_attr(raw_ctx, Activity::ConstNoNeed),
        ];
        let ret_activity_arr = [
            activity_attr(raw_ctx, Activity::ConstNoNeed),
            activity_attr(raw_ctx, Activity::ActiveNoNeed),
            activity_attr(raw_ctx, Activity::DupNoNeed),
            activity_attr(raw_ctx, Activity::Const),
            activity_attr(raw_ctx, Activity::Dup),
            activity_attr(raw_ctx, Activity::Active),
        ];

        create_autodiff_op(
            raw_ctx,
            "all_activity",
            &result_types,
            &[],
            &activity_arr,
            &ret_activity_arr,
            1,
            false,
            raw_loc,
        )
    };

    assert!(!autodiff_op.ptr.is_null());
    module
        .body()
        .append_operation(unsafe { melior::ir::Operation::from_raw(autodiff_op) });

    let text = module.as_operation().to_string();
    eprintln!("activity_attrs_cover_all_variants:\n{text}");
    for activity in [
        "enzyme_active",
        "enzyme_dup",
        "enzyme_const",
        "enzyme_dupnoneed",
        "enzyme_activenoneed",
        "enzyme_constnoneed",
    ] {
        assert!(
            text.contains(activity),
            "missing activity variant {activity}"
        );
    }
}