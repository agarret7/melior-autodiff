// Every Enzyme pass constructor builds a pass that runs.

use melior::{
    ir::Module,
    pass::{Pass, PassManager},
};
use melior_autodiff::{pass, utility::create_context};

type CreatePass = fn() -> Pass;

#[test]
fn every_pass_runs() {
    let constructors: &[(&str, CreatePass)] = &[
        ("enzyme", pass::create_differentiate_pass),
        ("enzyme-batch", pass::create_batch_pass),
        ("enzyme-diff-batch", pass::create_batch_diff_pass),
        ("enzyme-activity-opt", pass::create_enzyme_activity_opt_pass),
        ("enzyme-simplify-math", pass::create_mathematic_simplification_pass),
        ("remove-unnecessary-enzyme-ops", pass::create_remove_unused_enzyme_ops_pass),
        ("flatten-enzyme-caches", pass::create_flatten_enzyme_caches_pass),
        ("simplify-memref-cache", pass::create_simplify_memref_cache_pass),
        ("enzyme-batch-to-tensor", pass::create_enzyme_batch_to_tensor_pass),
        ("convert-enzyme-to-memref", pass::create_enzyme_ops_to_mem_ref_pass),
        ("add-to-op-to-split", pass::create_add_to_op_to_split_pass),
        ("split-multi-results", pass::create_split_multi_results_pass),
        ("inline-enzyme-regions", pass::create_inline_enzyme_into_region_pass),
        ("outline-enzyme-regions", pass::create_outline_enzyme_from_region_pass),
        ("hoist-enzyme-regions", pass::create_hoist_enzyme_from_region_pass),
        ("sink-checkpoint-views", pass::create_sink_checkpoint_views_pass),
        ("hoist-loop-allocations", pass::create_hoist_loop_allocations_pass),
        ("lower-affine-atomic-rmw", pass::create_lower_affine_atomic_rmw_pass),
        ("raise-llvm-ext", pass::create_raise_llvm_ext_pass),
        ("lower-llvm-ext", pass::create_lower_llvm_ext_pass),
    ];
    let ctx = create_context();
    for (name, create) in constructors {
        let pm = PassManager::new(&ctx);
        pm.add_pass(create());
        let mut module = Module::parse(&ctx, "module {}").unwrap();
        pm.run(&mut module).unwrap_or_else(|e| panic!("{name} failed: {e}"));
    }
}

#[test]
fn enzyme_lowering_runs() {
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    pass::add_enzyme_lowering(&pm);
    let mut module = Module::parse(&ctx, "module {}").unwrap();
    pm.run(&mut module).unwrap();
}

// `register_passes` makes every pass nameable in a textual pipeline.
#[test]
fn passes_are_registered_by_name() {
    melior_autodiff::utility::register_passes();
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    melior::utility::parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(enzyme, flatten-enzyme-caches, convert-enzyme-to-memref)",
    )
    .unwrap();
    let mut module = Module::parse(&ctx, "module {}").unwrap();
    pm.run(&mut module).unwrap();
}
