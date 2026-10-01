use melior::ir::{
    attribute::{BoolAttribute, FlatSymbolRefAttribute, IntegerAttribute},
    operation::OperationBuilder,
    r#type::IntegerType,
    BlockLike, Identifier, Location, Module, Type,
};
use melior_autodiff::Activity;

mod common;
use common::{activity_array_attr, setup_context};

#[test]
fn activity_attrs_cover_all_variants() {
    let ctx = setup_context();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);

    let result_types = [Type::float64(&ctx)];
    let autodiff_op = OperationBuilder::new("enzyme.autodiff", loc)
        .add_results(&result_types)
        .add_attributes(&[
            (
                Identifier::new(&ctx, "fn"),
                FlatSymbolRefAttribute::new(&ctx, "all_activity").into(),
            ),
            (
                Identifier::new(&ctx, "activity"),
                activity_array_attr(
                    &ctx,
                    &[
                        Activity::Active,
                        Activity::Dup,
                        Activity::Const,
                        Activity::DupNoNeed,
                        Activity::ActiveNoNeed,
                        Activity::ConstNoNeed,
                    ],
                )
                .into(),
            ),
            (
                Identifier::new(&ctx, "ret_activity"),
                activity_array_attr(
                    &ctx,
                    &[
                        Activity::ConstNoNeed,
                        Activity::ActiveNoNeed,
                        Activity::DupNoNeed,
                        Activity::Const,
                        Activity::Dup,
                        Activity::Active,
                    ],
                )
                .into(),
            ),
            (
                Identifier::new(&ctx, "width"),
                IntegerAttribute::new(IntegerType::new(&ctx, 64).into(), 1).into(),
            ),
            (
                Identifier::new(&ctx, "strong_zero"),
                BoolAttribute::new(&ctx, false).into(),
            ),
        ])
        .build()
        .unwrap();

    module.body().append_operation(autodiff_op);

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
