//! The generated sedan's render mesh (Collada) and materials.

use super::build::*;
use super::export::Flex;

pub struct Visual {
    pub flex: Vec<(Part, Flex)>,
    pub deform_groups: Vec<(String, Vec<usize>, f32)>,
    pub collada: String,
    pub materials: String,
}

pub fn build(_car: &Car) -> Visual {
    Visual { flex: Vec::new(), deform_groups: Vec::new(), collada: String::new(), materials: "{}".into() }
}
