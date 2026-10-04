// NN is replaced with the sample count by tests/impulse.rs before parsing.
module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %mean : tensor<2xui64>, tensor<f64>
  }
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64> {
    %half = arith.constant dense<-0.5> : tensor<f64>
    %d = arith.subf %x, %mean : tensor<f64>
    %z = arith.divf %d, %stddev : tensor<f64>
    %z2 = arith.mulf %z, %z : tensor<f64>
    %lp = arith.mulf %half, %z2 : tensor<f64>
    return %lp : tensor<f64>
  }
  func.func private @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func private @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<NNx1xf64> {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:9 = impulse.infer @model(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = NN }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<NNx1xf64>, tensor<NNx2xi1>, tensor<NNxf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0 : tensor<NNx1xf64>
  }
  func.func @run(%seed0: i64, %seed1: i64, %mean: f64, %stddev: f64, %out: memref<NNx1xf64>) attributes { llvm.emit_c_interface } {
    %u0 = builtin.unrealized_conversion_cast %seed0 : i64 to ui64
    %u1 = builtin.unrealized_conversion_cast %seed1 : i64 to ui64
    %rng = tensor.from_elements %u0, %u1 : tensor<2xui64>
    %m = tensor.from_elements %mean : tensor<f64>
    %sd = tensor.from_elements %stddev : tensor<f64>
    %samples = func.call @hmc(%rng, %m, %sd) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> tensor<NNx1xf64>
    bufferization.materialize_in_destination %samples in writable %out : (tensor<NNx1xf64>, memref<NNx1xf64>) -> ()
    return
  }
}

// Differentiation only (hmc_gaussian_differentiates): every autodiff region is outlined and
// differentiated, and the log-density gradient goes through the dynamic_slice adjoint.
// DIFF-NOT:     enzyme.autodiff_region
// DIFF-NOT:     enzyme.autodiff @
// DIFF-LABEL: func.func private @diffehmc_to_diff0(
// DIFF:         impulse.dynamic_update_slice
