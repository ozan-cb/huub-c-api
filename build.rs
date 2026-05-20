// Generate include/huub.h from the `extern "C"` surface in src/lib.rs.
//
// We commit the generated header into include/ so C++ consumers can
// include it without running cargo. The build.rs regenerates it on
// every cargo build to keep it in sync with the Rust source.

use std::{env, path::PathBuf};

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let header_path = crate_dir.join("include").join("huub.h");

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("read cbindgen.toml");

    // Don't fail the build if the header can't be regenerated (e.g.
    // running under a sandbox with no write access to include/) — but
    // do surface a warning so it's noticed in CI.
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
        }
        Err(e) => {
            println!("cargo:warning=cbindgen failed: {e}");
        }
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
