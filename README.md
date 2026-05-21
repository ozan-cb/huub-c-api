# huub-c-api

A C ABI wrapper around the [Huub](https://github.com/huub-solver/huub) CP solver, exposing a stable `extern "C"` surface that can be called from C, C++, or any language with a C FFI.

## Features

- Opaque model handle (`HuubModel*`) with full lifecycle management
- Integer decision variables with arbitrary `[lb, ub]` domains
- Boolean decision variables
- Linear constraints (`<=`, `<`, `>=`, `>`, `==`, `!=`)
- Single-worker satisfaction solve with optional wall-clock time limit
- Thread-local last-error string (no global state)
- Panic safety: Rust panics are caught at every FFI boundary and reported as error codes

## Building

Requires Rust 1.85+ (edition 2024).

```sh
cargo build --release
```

This produces:
- `libhuub_c_api.a` — static library
- `libhuub_c_api.so` / `libhuub_c_api.dylib` — dynamic library
- `include/huub.h` — generated C header (also committed for convenience)

The header is regenerated automatically on each build via `build.rs` and `cbindgen`.

## Usage

Include `include/huub.h` and link against the built library.

```c
#include "huub.h"
#include <stdio.h>

int main(void) {
    HuubModel *m = huub_model_new();

    int32_t x = huub_model_new_int_var(m, 0, 10);
    int32_t y = huub_model_new_int_var(m, 0, 10);

    // x + y == 7
    int32_t ids[2] = {x, y};
    int64_t cs[2]  = {1, 1};
    huub_model_add_linear(m, ids, cs, 2, HUUB_REL_OP_EQ, 7);

    HuubResult r = huub_model_solve(m, 10.0);
    if (r == HUUB_RESULT_SATISFIED) {
        int64_t xv, yv;
        huub_model_value_int(m, x, &xv);
        huub_model_value_int(m, y, &yv);
        printf("x=%lld y=%lld\n", (long long)xv, (long long)yv);
    }

    huub_model_free(m);
    return 0;
}
```

## API reference

See [`include/huub.h`](include/huub.h) for the full API with Doxygen-style documentation on every entry point.

### Error handling

Every function that can fail either returns a sentinel value (`NULL` for handles, `-1` for variable IDs, `HUUB_RESULT_ERROR` for result codes) and records a human-readable message retrievable via `huub_last_error()`. The error string is valid until the next FFI call on the same thread.

### Thread safety

Handles are **not thread-safe**. Create and use each `HuubModel*` on a single thread.

## License

This crate is MIT licensed. See [LICENSE](LICENSE).

The [Huub](https://github.com/huub-solver/huub) solver dependency is licensed under [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/). MPL-2.0 is a file-level copyleft license; the Huub source files remain under MPL-2.0, while this wrapper's source files are separately MIT licensed. Binary distributions that include compiled Huub code must make Huub's source available per the MPL-2.0 terms.
