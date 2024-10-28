// Config that does not use MPI
// Used for WASM

use std::fmt::Debug;
use arith::{Field, FieldSerde};
use transcript::{FiatShamirHash, Transcript, TranscriptInstance};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct MPIConfig {}

/// MPI toolkit:
impl MPIConfig {
    const ROOT_RANK: usize = 0;
    const ROOT_SIZE: usize = 1;
    // OK if already initialized, mpi::initialize() will return None
    #[allow(static_mut_refs)]
    pub fn init() {
        // NOTHING
    }

    #[inline]
    pub fn finalize() {
        // NOTHING
    }

    #[allow(static_mut_refs)]
    pub fn new() -> Self {
        Self::init();
        Self {}
    }

    #[allow(clippy::collapsible_else_if)]
    pub fn gather_vec<F: Field>(&self, local_vec: &Vec<F>, global_vec: &mut Vec<F>) {
        *global_vec = local_vec.clone()
    }

    /// Root process broadcase a value f into all the processes
    #[inline]
    pub fn root_broadcast<F: Field>(&self, _f: &mut F) {
        // NOTHING
    }

    /// sum up all local values
    #[inline]
    pub fn sum_vec<F: Field>(&self, local_vec: &Vec<F>) -> Vec<F> {
        local_vec.clone()
    }

    /// coef has a length of mpi_world_size
    #[inline]
    pub fn coef_combine_vec<F: Field>(&self, local_vec: &Vec<F>, _coef: &[F]) -> Vec<F> {
        // Warning: literally, it should be coef[0] * local_vec
        // but coef[0] is always one in our use case of self.world_size = 1
        local_vec.clone()
    }

    #[inline(always)]
    pub fn world_size(&self) -> usize {
        Self::ROOT_SIZE
    }

    #[inline(always)]
    pub fn world_rank(&self) -> usize {
        Self::ROOT_RANK
    }

    #[inline(always)]
    pub fn is_root(&self) -> bool {
        true
    }

    #[inline(always)]
    pub fn barrier(&self) {
        // NOTHING
    }

    /// broadcast root transcript state. incurs an additional hash if self.world_size > 1
    #[inline]
    pub fn transcript_sync_up<H: FiatShamirHash>(&self, _transcript: &mut TranscriptInstance<H>) {
        // NOTHING
    }

    /// Transcript IO for MPI
    #[inline]
    pub fn transcript_io<F, H>(&self, ps: &[F], transcript: &mut TranscriptInstance<H>) -> F
    where
        F: Field + FieldSerde,
        H: FiatShamirHash,
    {
        assert!(ps.len() == 3 || ps.len() == 4); // 3 for x, y; 4 for simd var
        for p in ps {
            transcript.append_field_element::<F>(p);
        }
        let mut r = transcript.generate_challenge::<F>();
        self.root_broadcast(&mut r);
        r
    }
}