// Every Enzyme pass this crate adds by name is registered under that name and runs.

use melior::{ir::Module, pass::PassManager};
use melior_autodiff::{pass, utility::create_context};

type AddPass = fn(&PassManager) -> Result<(), melior::Error>;

#[test]
fn pipeline_passes_are_registered() {
    let adds: &[(&str, AddPass)] = &[
        ("enzyme-simplify-math", pass::add_mathematic_simplification_pass),
        ("enzyme-activity-opt", pass::add_enzyme_activity_opt_pass),
        ("flatten-enzyme-caches", pass::add_flatten_enzyme_caches_pass),
        ("simplify-memref-cache", pass::add_simplify_memref_cache_pass),
        ("enzyme-batch-to-tensor", pass::add_enzyme_batch_to_tensor_pass),
        ("add-to-op-to-split", pass::add_add_to_op_to_split_pass),
        ("split-multi-results", pass::add_split_multi_results_pass),
        ("inline-enzyme-regions", pass::add_inline_enzyme_into_region_pass),
        ("outline-enzyme-regions", pass::add_outline_enzyme_from_region_pass),
        ("hoist-enzyme-regions", pass::add_hoist_enzyme_from_region_pass),
        ("sink-checkpoint-views", pass::add_sink_checkpoint_views_pass),
        ("hoist-loop-allocations", pass::add_hoist_loop_allocations_pass),
        ("lower-affine-atomic-rmw", pass::add_lower_affine_atomic_rmw_pass),
        ("raise-llvm-ext", pass::add_raise_llvm_ext_pass),
        ("lower-llvm-ext", pass::add_lower_llvm_ext_pass),
    ];
    let ctx = create_context();
    for (name, add) in adds {
        let pm = PassManager::new(&ctx);
        add(&pm).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut module = Module::parse(&ctx, "module {}").unwrap();
        pm.run(&mut module).unwrap_or_else(|e| panic!("{name} failed: {e}"));
    }
}

#[test]
fn enzyme_lowering_runs() {
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    pass::add_enzyme_lowering(&pm).unwrap();
    let mut module = Module::parse(&ctx, "module {}").unwrap();
    pm.run(&mut module).unwrap();
}
