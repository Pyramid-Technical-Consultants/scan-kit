//! Chamber spacing used by Dose Volume.

/// IC2 is the plot origin. IC1 sits 100 mm downstream.
pub const IC2_Z_MM: f32 = 0.0;
pub const IC1_Z_MM: f32 = 100.0;
pub const IC_SEP_MM: f32 = IC1_Z_MM - IC2_Z_MM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_025_chambers_are_100_mm_apart() {
        assert_eq!(IC2_Z_MM, 0.0);
        assert_eq!(IC1_Z_MM, 100.0);
        assert_eq!(IC_SEP_MM, 100.0);
    }
}
