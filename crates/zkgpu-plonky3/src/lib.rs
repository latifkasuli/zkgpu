// Adapter between zkgpu and Plonky3 field/DFT interfaces.
//
// This crate will provide:
// - Conversion between p3_baby_bear::BabyBear and zkgpu_babybear::BabyBear
// - A TwoAdicSubgroupDft implementation backed by GPU NTT
//
// Not yet implemented — waiting for the GPU NTT to be validated first.
