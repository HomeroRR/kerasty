//! Keras for Rust with support for Web Assembly.
//!
//! ## Features
//!
//! - Candle backend
//!
//! # Roadmap of Supported Layers

//! |    Layer   | State |                        Example                            |
//! |------------|-------|-----------------------------------------------------------|
//! |    Dense   |&#9989;| [add](https://docs.rs/kerasty/latest/kerasty/fn.add.html) |

/// Adds two numbers.
///
/// # Examples
///
/// ```
/// let result = kerasty::add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
