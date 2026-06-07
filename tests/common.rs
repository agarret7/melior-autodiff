use melior::{Context, ir::Module, dialect::DialectRegistry, utility::register_all_dialects};
use melior_autodiff::{enzymeRegisterDialectExtensions, enzyme_dialect_handle};
  use mlir_sys::mlirDialectHandleLoadDialect;


pub fn setup_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    // Register Enzyme's external AD models for arith, func, scf, etc.
    // before the context is created so extensions fire as each dialect
    // (including BuiltinDialect) is loaded for the first time.
    unsafe { enzymeRegisterDialectExtensions(registry.to_raw()) };
    let ctx = Context::new_with_registry(&registry, false);
    ctx.load_all_available_dialects();
    unsafe {
        let handle = enzyme_dialect_handle();
        mlirDialectHandleLoadDialect(handle, ctx.to_raw());
    }
    ctx
}

pub fn parse_square_autodiff_module(ctx: &Context) -> Module<'_> {
    Module::parse(
        ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dr) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %r : f64
  }
}
"#,
    )
    .expect("failed to parse test module")
}

pub fn parse_tensor_fwddiff_module(ctx: &Context) -> Module<'_> {
    Module::parse(
        ctx,
        r#"
module {
  func.func @sin_tensor(%x: tensor<2xf64>) -> tensor<2xf64> {
    %y = math.sin %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }

  func.func @dsin_tensor(%x: tensor<2xf64>, %dx: tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @sin_tensor(%x, %dx) {
      activity=[#enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}
"#,
    )
    .expect("failed to parse tensor fwddiff test module")
}

pub fn parse_square_jacobian_module(ctx: &Context) -> Module<'_> {
    Module::parse(
        ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.jacobian @square(%x, %dr) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %r : f64
  }
}
"#,
    )
    .expect("failed to parse jacobian test module")
}

pub fn parse_tensor_autodiff_module(ctx: &Context) -> Module<'_> {
    Module::parse(
        ctx,
        r#"
module {
  func.func @square_tensor(%x: tensor<2xf64>) -> tensor<2xf64> {
    %y = arith.mulf %x, %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }

  func.func @dsquare_tensor(%x: tensor<2xf64>, %dr: tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.autodiff @square_tensor(%x, %dr) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}
"#,
    )
    .expect("failed to parse tensor autodiff test module")
}