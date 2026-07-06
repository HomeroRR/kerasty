#!/usr/bin/env bash
#
# Kerasty development helper.
#
#   ./dev.sh <command>
#
# Commands:
#   fmt      Format the code with rustfmt
#   fmt-check  Check formatting without modifying files
#   lint     Run clippy, denying all warnings
#   test     Run the test suite
#   build    Build the crate (native)
#   doc      Build and open the API docs
#   wasm     Build for wasm32-unknown-unknown (inference target)
#   check    fmt-check + lint + test  (what CI should run)
#   all      fmt + lint + test + build + wasm
#
# With no argument, runs `check`.

set -euo pipefail

# Make cargo/rustup available even in minimal shells.
export PATH="$HOME/.cargo/bin:$PATH"

WASM_TARGET="wasm32-unknown-unknown"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }

cmd_fmt()       { info "rustfmt";            cargo fmt; }
cmd_fmt_check() { info "rustfmt --check";    cargo fmt --check; }
cmd_lint()      { info "clippy -D warnings"; cargo clippy --all-targets -- -D warnings; }
cmd_test()      { info "cargo test";         cargo test; }
cmd_build()     { info "cargo build";        cargo build; }
cmd_doc()       { info "cargo doc";          cargo doc --no-deps --open; }

cmd_wasm() {
    info "wasm build ($WASM_TARGET)"
    if ! rustup target list --installed | grep -q "$WASM_TARGET"; then
        info "installing target $WASM_TARGET"
        rustup target add "$WASM_TARGET"
    fi
    cargo build --target "$WASM_TARGET"
}

cmd_check() {
    cmd_fmt_check
    cmd_lint
    cmd_test
}

cmd_all() {
    cmd_fmt
    cmd_lint
    cmd_test
    cmd_build
    cmd_wasm
}

main() {
    local command="${1:-check}"
    case "$command" in
        fmt)        cmd_fmt ;;
        fmt-check)  cmd_fmt_check ;;
        lint)       cmd_lint ;;
        test)       cmd_test ;;
        build)      cmd_build ;;
        doc)        cmd_doc ;;
        wasm)       cmd_wasm ;;
        check)      cmd_check ;;
        all)        cmd_all ;;
        *)
            echo "Unknown command: $command" >&2
            echo "Run './dev.sh' with one of: fmt, fmt-check, lint, test, build, doc, wasm, check, all" >&2
            exit 1
            ;;
    esac
}

main "$@"
