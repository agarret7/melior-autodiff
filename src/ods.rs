//! Enzyme's ops and their builders, generated from its TableGen definitions the way
//! `melior::dialect::ods` generates upstream dialects.

mod generated {
    melior::dialect! {
        name: "enzyme",
        files: ["EnzymeOps.td"],
        include_directory_env_vars: ["ENZYME_DIALECT_DIR"],
    }
}

pub use generated::enzyme::*;
