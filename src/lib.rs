//
//  Copyright 2024 Jonathan L Bertoni
//
//  This code is available under the Berkeley 2-Clause, Berkeley 3-clause,
//  and MIT licenses.
//

//! BuddyPool implements a simple buddy-system allocator.  It does not assume
//! that the memory (or resource) being allocated is accessible, and uses only
//! its own data structures for metadata.  It thus can be used, for example to
//! allocate I/O memory that is not directly accessible to the CPU.  As part of
//! this approach, BuddyPool does not contain any unsafe code, using Vec structures
//! with integers for linking lists.  This approach does consume more memory than
//! some other approaches.
//!
//! ## Types
//!
//! * BuddyPool
//!    * This struct is the type for a memory pool.
//! * BuddyConfig
//!    * This struct is used to pass configuration parameters to the pool constructor.
//! * BuddyStats
//!    * The pool implementation keeps some statistics on operations.  This struct is
//!      used to return them to the user.
//! * BuddyError
//!    * Pool functions use this type to return information on errors.
//! * Index
//!    * The allocator uses this type for indices.  Types u8, u16, u32, and usize
//!      have been tested.  Smaller types will limit the maximum pool size, but save
//!      on overhead.
//!
//! ## Example
//!
//!```
//!        use buddy_pool::BuddyPool;
//!        use buddy_pool::BuddyConfig;
//!        use buddy_pool::BuddyErrorType;
//!        use std::alloc::alloc;
//!        use std::alloc::Layout;
//!
//!        // Allocate some memory for testing.
//!
//!        let bytes: usize = 2 * 1024 * 1024;
//!
//!        let alloc_mem =
//!            unsafe {
//!                alloc(Layout::array::<u8>(bytes).unwrap())
//!            } as usize;
//!
//!        // Get the configuration parameters ready.
//!
//!        let config =
//!            BuddyConfig {
//!                base:       alloc_mem  ,        // the base address from libc::malloc
//!                size:       bytes,              // the size of the memory region in bytes
//!                min_alloc:  1,                  // the minimum allocation size allowed
//!                max_alloc:  17 * 1024,          // the largest allocation that will be allowed
//!                min_buddy:  8192,               // the minimum size buddy block that will be created
//!            };
//!
//!        // Construct the pool and try an alloc/free pair.
//!
//!        let mut pool    = BuddyPool::new(config).unwrap();
//!        let     address = pool.alloc(config.max_alloc).unwrap();
//!
//!        assert!(address >= config.base && address < config.base + config.size);
//!
//!        pool.free(address).unwrap();
//!
//!        // Try a double free.
//!
//!        let result = pool.free(address);
//!
//!        assert!(result.is_err());
//!        assert!(result.unwrap_err().error_type() == BuddyErrorType::FreeingFreeMemory);
//!
//!        // Try an undersized allocation.
//!
//!        let result = pool.alloc(config.min_alloc - 1);
//!
//!        assert!(result.is_err());
//!        assert!(result.unwrap_err().error_type() == BuddyErrorType::UndersizeAlloc);
//!
//!        // Try an oversized allocation.
//!
//!        let result = pool.alloc(config.max_alloc + 1);
//!
//!        assert!(result.is_err());
//!        assert!(result.unwrap_err().error_type() == BuddyErrorType::OversizeAlloc);
//!
//!        // Now allocate until the pool is empty.  Keep track
//!        // of the allocation count to see that it matches
//!        // the configuration.
//!
//!        let mut allocs = 0;
//!
//!        loop {
//!            let result = pool.alloc(config.min_alloc);
//!
//!            if result.is_err() {
//!                let error = result.unwrap_err();
//!
//!                assert!(error.error_type() == BuddyErrorType::OutOfMemory);
//!                break;
//!            }
//!
//!            allocs += 1;
//!        }
//!
//!        assert!(allocs == config.size / config.min_buddy);
//!

use std::fmt;
use std::fmt::Display;

/// This type can be set to u8, u16, u32, or usize, as needed to
/// match the pool size.  The size limit of a pool in bytes is
/// usize::MAX / 2 or (Index::MAX + 1) * min_size, whichever
/// is smaller.  For example, a min_size parameter of 1024 bytes
/// using u32 indices sets a limit of 4 terabytes per memory pool.

pub type Index = u32;

/// Specifies the configuration parameters for a pool.

#[derive(Clone, Copy)]
pub struct BuddyConfig {
    pub base:       usize,
    pub size:       usize,

    pub min_alloc:  usize,
    pub min_buddy:  usize,
    pub max_alloc:  usize,
}

/// Defines the pool type itself.

pub struct BuddyPool {
    base:           usize,
    end:            usize,
    min_size:       usize,  // minimum buddy size to allocate
    max_size:       usize,  // maximum buddy size to allocate
    min_alloc:      usize,  // minimum size for an alloc() call
    max_alloc:      usize,  // maximum size for an alloc() call
    allocs:         usize,
    frees:          usize,
    out_of_mem:     usize,  // out-of-memory failures
    freeing_free:   usize,  // free() called with invalid address
    splits:         usize,  // buddy block splits
    merges:         usize,  // buddy block merges
    log2_min_size:  usize,

    leaves:         Vec<Leaf>,
    free_lists:     Vec<ListHead>,
    // locked_lists:  Vec<ListHead>,
}

impl Display for BuddyPool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BuddyPool: at {:#x}", self.base)
    }
}

impl fmt::Debug for BuddyPool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BuddyPool: at {:#x}", self.base)
    }
}

/// Contains the results of a query for statistics on the pool.

pub struct BuddyStats {
    pub min_size:     usize,
    pub max_size:     usize,
    pub allocs:       usize,
    pub frees:        usize,
    pub out_of_mem:   usize,
    pub freeing_free: usize,
    pub splits:       usize,
    pub merges:       usize,

    pub freelist_stats: Vec<ListStats>,
}

/// Contains the statistics returned by a single list.
/// Currently, the only lists are the free lists.

pub struct ListStats {
    pub dequeues:    usize,
    pub enqueues:    usize,
    pub removes:     usize,
    pub size:        usize,
}

/// Contains the error information for failed requests.

#[derive(Clone, Debug)]
pub struct BuddyError {
    error_type: BuddyErrorType,
    message:    String,
}

impl BuddyError {
    pub fn new(error_type: BuddyErrorType, message: String) -> BuddyError {
        BuddyError { error_type, message }
    }

    pub fn new_str(error_type: BuddyErrorType, message: &str) -> BuddyError {
        let message = message.to_string();

        BuddyError { error_type, message }
    }

    pub fn error_type(&self) -> BuddyErrorType {
        self.error_type
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Defines the error codes for all operations.

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BuddyErrorType {
    InvalidParameter,
    InvalidSize,
    UndersizeAlloc,
    OversizeAlloc,
    OutOfMemory,
    InvalidAddress,
    AllocAddressMisalignment,
    AllocSizeMismatch,
    FreeingFreeMemory,
    InvalidListId,
    InvalidLeafId,
    IndexTypeOverflow,
    ListIdMismatch,
    AttachedLeaf,
    UnattachedLeaf,
    EmptyList,
    EmptyPool,
}

impl Display for BuddyErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Defines the per-leaf information.  Each leaf corresponds to
/// "min_buddy" bytes in the pool.  The current_index value
/// corresponds to the free list index for the current size of
/// the block.  As the block is split and merged, this index
/// is updated.

#[derive(Debug)]
struct Leaf {
    state:          LeafState,
    current_index:  Index,
    list_id:        u32,
    next:           Index,
    prev:           Index,
}

impl Leaf {
    fn new(state: LeafState, current_index: Index) -> Leaf {
        let list_id = u32  ::MAX;
        let next    = Index::MAX;
        let prev    = Index::MAX;

        Leaf { state, current_index, next, prev, list_id }
    }

    fn current_size(&self, base_index: usize) -> usize {
        2_usize.pow((self.current_index + base_index as Index) as u32)
    }

    fn invalidate_list_info(&mut self) {
        self.next    = Index::MAX;
        self.prev    = Index::MAX;
        self.list_id = u32  ::MAX;
    }

    fn is_off_lists(&self) -> bool {
            self.next    == Index::MAX
        &&  self.prev    == Index::MAX
        &&  self.list_id == u32  ::MAX
    }

    fn is_on_list(&self, list_id: u32, max_leaf: usize) -> bool {
            (self.next as usize)  <  max_leaf
        &&  (self.prev as usize)  <  max_leaf
        &&  self.list_id         == list_id
    }

    fn is_list_consistent(&self, max_list_id: u32, max_leaf: usize) -> bool {
        if self.is_off_lists() {
            true
        } else {
                (self.next as usize) < max_leaf
            &&  (self.prev as usize) < max_leaf
            &&  self.list_id        < max_list_id
        }
    }
}

/// Defines the list head structure.  Currently, this
/// is used only for the free lists.

#[derive(Debug)]
struct ListHead {
    id:    u32,
    size:  usize,
    first: usize,

    // Usage statistics

    dequeues:    usize,
    enqueues:    usize,
    removes:     usize,
}

impl ListHead {
    pub fn new(id: u32) -> ListHead {
        let size      = 0;
        let first     = usize::MAX;
        let enqueues  = 0;
        let dequeues  = 0;
        let removes   = 0;

        ListHead { id, size, first, enqueues, dequeues, removes }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LeafState {
    Free,
    Allocated,
    Merged,
    Wanted,
    WantedHead,
    Locked,
    LockedHead,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BuddyAction {
    Split,
    Merge,
}

impl Display for LeafState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl BuddyPool {
    /// Creates a new pool.

    pub fn new(config: BuddyConfig) -> Result<BuddyPool, BuddyError> {
        // Check that we can round max_size up.

        if config.max_alloc == 0 || config.max_alloc > usize::MAX / 2 {
            let message =
                format!
                (
                    "The max_alloc value ({}) is out of the range [1, usize::MAX / 2]",
                    config.max_alloc
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if config.min_alloc > usize::MAX / 2 {
            let message =
                format!
                (
                    "The min_alloc value ({}) is out of the range [0, usize::MAX / 2]",
                    config.min_alloc
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if config.min_buddy == 0 || config.min_buddy > usize::MAX / 2 {
            let message =
                format!
                (
                    "The min_buddy value ({}) is out of the range [1, usize::MAX / 2]",
                    config.min_buddy
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        // Convert min_buddy to the smallest power of 2 <= min_buddy.
        // Convert max_buddy to the smallest power of 2 >= max_alloc.

        let min_size  = 2_usize.pow(config.min_buddy.ilog2());
        let max_size  = config.max_alloc.next_power_of_two();
        let pool_size = Self::truncate(config.size, min_size);

        if pool_size == 0 || pool_size > usize::MAX / 2 {
            let message =
                format!
                (
                    "The pool size ({}) is out of the range [1, usize::MAX / 2]",
                    pool_size
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        let max_base = (usize::MAX / 2) - pool_size - min_size;

        if config.base > max_base {
            let message =
                format!
                (
                    "The base ({}) is out of the range [0, {}]",
                    config.base,
                    max_base
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if config.max_alloc > pool_size {
            let message =
                format!
                (
                    "The max allocation size ({}) is larger than the pool size ({}).",
                    config.max_alloc,
                    pool_size
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if config.min_alloc > config.max_alloc {
            let message =
                format!
                (
                    "The min allocation size ({}) is larger than the max allocation size ({}).",
                    config.min_alloc,
                    config.max_alloc
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if config.min_buddy > max_size {
            let message =
                format!
                (
                    "The min buddy size ({}) is larger than the max buddy size ({}).",
                    config.min_buddy,
                    max_size
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        if max_size > pool_size {
            let message =
                format!
                (
                    "The buddy size ({}) of the max allocation size ({}) exceeds the pool size ({}).",
                    max_size,
                    config.max_alloc,
                    pool_size
                );

            let error = BuddyError::new(BuddyErrorType::InvalidParameter, message);
            return Err(error);
        }

        // We might be using u32 or u16 as the index, so check for possible
        // overflow.

        // Compute the max index possible for leaves.

        let max_index = (pool_size / min_size) - 1;

        if max_index > Index::MAX as usize {
            let message =
                format!
                (
                    "That Index type (max {}) is too small for the pool size ({}) with min_buddy {}.",
                    Index::MAX,
                    pool_size,
                    min_size
                );

            return Err(BuddyError::new(BuddyErrorType::IndexTypeOverflow, message));
        }

        let     list_count   = max_size.ilog2() - min_size.ilog2() + 1;
        let mut free_lists   = Vec::with_capacity(list_count as usize);
        // let mut locked_lists = Vec::with_capacity(list_count as usize);

        // Initialize each list with a unique id.

        for i in 0..list_count {
            free_lists  .push(ListHead::new(i             ));
            // locked_lists.push(ListHead::new(i + list_count));
        }

        let     leaf_count  = pool_size / min_size;
        let mut leaves      = Vec::with_capacity(leaf_count);

        for _i in 0..leaf_count {
            let leaf = Leaf::new(LeafState::Free, 0);

            leaves.push(leaf);
        }

        Self::setup_leaves(&mut leaves, &mut free_lists, min_size, max_size)?;

        // Now we are ready to build the pool struct.

        let base           = config.base;
        let end            = base + pool_size;
        let min_alloc      = config.min_alloc;
        let max_alloc      = config.max_alloc;
        let allocs         = 0;
        let frees          = 0;
        let out_of_mem     = 0;
        let freeing_free   = 0;
        let splits         = 0;
        let merges         = 0;
        let log2_min_size  = min_size.ilog2() as usize;

        Ok(BuddyPool {
            base,
            end,
            min_alloc,
            max_alloc,
            min_size,
            max_size,
            allocs,
            frees,
            out_of_mem,
            freeing_free,
            splits,
            merges,
            log2_min_size,
            leaves,
            free_lists,
            // locked_lists,
        })
    }

    fn truncate(size: usize, min_size: usize) -> usize {
        (size / min_size) * min_size
    }

    // This help function initializes the leaves array and the free lists.

    fn setup_leaves(leaves: &mut [Leaf], freelists: &mut [ListHead], min_size: usize, max_size: usize)
            -> Result<(), BuddyError> {

        if leaves.is_empty() {
            let message = "The pool contains no leaves.";

            return Err(BuddyError::new_str(BuddyErrorType::EmptyPool, message));
        }

        let mut current_leaf    = 0_usize;
        let mut current_size    = max_size;
        let mut remaining       = leaves.len() * min_size;
        let mut freelist_index  = freelists.len();  // Start at one past the end...

        // Check that the parameters make sense.  "freelist_index" currently
        // points just past the end of the free list Vec.

        assert!(max_size == 2_usize.pow(freelist_index as u32 - 1) * min_size);

        // Now create the free lists by marking all memory as free.  Start
        // with the largest allocation size, and make as many free blocks
        // as possible, then move downward in size until all the  memory is
        // on some freelist.

        while remaining >= min_size && freelist_index > 0 {
            // Move to the next freelist index.  Start at len() to avoid
            // possible underflow when decrementing but allow the
            // (hopefully redundant) freelist_index check for the loop.

            freelist_index -= 1;

            let leaves_per_unit = current_size / min_size;

            // Make as many blocks of this size as possible.

            while remaining >= current_size {
                leaves[current_leaf].current_index = freelist_index as Index;

                // Mark all leaves other than the first in this block as merged.

                for i in 1..leaves_per_unit {
                    leaves[current_leaf + i].state         = LeafState::Merged;
                    leaves[current_leaf + i].current_index = 0;
                }

                Self::enqueue(&mut freelists[freelist_index], leaves, current_leaf).unwrap();

                if current_leaf <= usize::MAX - leaves_per_unit {
                    current_leaf += leaves_per_unit;
                }

                // We have consumed "current_size" bytes, so update the remaining
                // free space size.

                remaining -= current_size;
            }

            current_size /= 2;
        }

        Ok(())
    }

    /// Allocates memory.

    pub fn alloc(&mut self, bytes: usize) -> Result<usize, BuddyError> {
        if bytes > self.max_alloc {
            let message =
                format!
                (
                    "That allocation size ({}) is greater than the pool alloc limit ({}).",
                    bytes,
                    self.max_alloc
                );

            return Err(BuddyError::new(BuddyErrorType::OversizeAlloc, message));
        }

        if bytes < self.min_alloc {
            let message =
                format!
                (
                    "That allocation size ({}) is less than the pool alloc limit ({}).",
                    bytes,
                    self.min_alloc
                );

            return Err(BuddyError::new(BuddyErrorType::UndersizeAlloc, message));
        }

        self.allocs += 1;

        // Get the buddy index for this size and check whether
        // anything that size is free.

        let result_list = self.compute_list_id(bytes).unwrap();
        let free_leaf   = self.dequeue_free(result_list).unwrap();

        // If we found a free block, we're done.

        if let Some(free_leaf) = free_leaf {
            let leaf_limit = self.leaves.len();
            let leaf       = &mut self.leaves[free_leaf];

            assert!(leaf.is_off_lists());
            assert!(leaf.is_list_consistent(self.free_lists.len() as u32, leaf_limit));

            let current_size = leaf.current_size(self.log2_min_size);

            assert!(bytes <= self.min_size || current_size >= bytes && current_size < 2 * bytes);
            assert!(leaf.state == LeafState::Free);

            leaf.state = LeafState::Allocated;
            return self.to_address(free_leaf);
        }

        // Okay, we need to split a larger block if we can find one.

        let mut split_list = self.free_lists.len();

        for i in result_list + 1..self.free_lists.len() {
            if self.free_lists[i].size > 0 {
                split_list = i;
                break;
            }
        }

        // If there's no block large enough available, return an error.

        if split_list == self.free_lists.len() {
            self.out_of_mem += 1;

            let message =
                format!("The pool has no memory available for that request size ({})", bytes);
            return Err(BuddyError::new(BuddyErrorType::OutOfMemory, message));
        }

        // Get the index of our result leaf, and dequeue that block from the
        // free list.

        let result_leaf = self.free_lists[split_list].first;

        self.dequeue_free(split_list).unwrap();

        loop {
            self.split_buddy(result_leaf);
            split_list -= 1;
            assert!(self.free_lists[split_list].size == 1);
            assert!(self.leaves[result_leaf].current_index == split_list as Index);

            // Check whether we have a block of the correct size.  If so,
            // we're done.

            if split_list == result_list {
                break;
            }
        }

        let leaf = &mut self.leaves[result_leaf];

        leaf.state         = LeafState::Allocated;
        leaf.current_index = result_list as Index;

        let current_size = leaf.current_size(self.log2_min_size);

        assert!(bytes <= self.min_size || current_size >= bytes && current_size < 2 * bytes);
        self.to_address(result_leaf)
    }

    fn split_buddy(&mut self, leaf_index: usize) {
        assert!(leaf_index < self.leaves.len());
        assert!(self.leaves[leaf_index].current_index > 0);

        let lists         = self.free_lists.len() as Index;
        let leaf          = &self.leaves[leaf_index];
        let current_list  = leaf.current_index;
        let next_list     = current_list - 1;
        let buddy_index   = self.buddy_id(leaf_index, BuddyAction::Split).unwrap();

        // Check that the state makes sense and put the buddy on
        // the appropriate free list.

        assert!(current_list < lists);
        assert!(leaf.state == LeafState::Free);
        assert!(leaf.is_off_lists());
        assert!(self.leaves[buddy_index].state == LeafState::Merged);

        self.enqueue_free(next_list as usize, buddy_index).unwrap();

        // Now set the current size of the first half of the block.

        self.leaves[leaf_index].current_index = next_list;
        self.splits += 1;
    }

    pub fn dump_address(&self, address: usize) -> String {
        if address < self.base || address >= self.end {
            return format!("{:#x} => invalid address", address);
        }

        let offset  = address - self.base;
        let index   = offset / self.min_size;
        let state   = self.leaves[index].state;
        let size    = self.leaves[index].current_size(self.log2_min_size);
        let aligned = offset % self.min_size == 0;

        format!("{:#x} => state {}, size {}, aligned = {}",
            address, state, size, aligned)
    }

    /// Frees a previously-made allocation.

    pub fn free(&mut self, area: usize) -> Result<(), BuddyError> {
        if area < self.base || area >= self.end {
            let message =
                format!
                (
                    "That address ({:#x}) is out of range ([{:#x}, {:#x}])",
                    area,
                    self.base,
                    self.end
                );

            return Err(BuddyError::new(BuddyErrorType::InvalidAddress, message));
        }

        let area_offset = area - self.base;
        let min_size    = self.min_size;

        if area_offset % min_size != 0 {
            let closest = area_offset.next_multiple_of(min_size) - min_size;

            let message =
                format!
                (
                    "That address ({:#x}) is incorrect (closest buddy {:#x})",
                    area,
                    self.base + closest
                );

            return Err(BuddyError::new(BuddyErrorType::InvalidAddress, message));
        }

        let leaf_id = area_offset / min_size;
        let list_id = self.leaves[leaf_id].current_index;
        let state   = self.leaves[leaf_id].state;

        if !self.is_allocated(state) {
            self.freeing_free += 1;

            let message =
                format!
                (
                    "That address ({:#x}) is not allocated ({}).",
                    area,
                    state
                );

            return Err(BuddyError::new(BuddyErrorType::FreeingFreeMemory, message));
        }

        self.frees += 1;

        let mut current_leaf = leaf_id;
        let mut current_list = list_id;

        loop {
            assert!(self.leaves[current_leaf].current_index == current_list);

            let result = self.try_merge_buddies(current_leaf);

            if result.is_none() {
                self.enqueue_free(current_list as usize, current_leaf).unwrap();
                break;
            }

            current_leaf  = result.unwrap();
            current_list += 1;
        }

        Ok(())
    }

    fn is_mergeable(&self, state: LeafState) -> bool {
            state == LeafState::Free
        ||  state == LeafState::Locked
        ||  state == LeafState::LockedHead
    }

    fn is_allocated(&self, state: LeafState) -> bool {
            state == LeafState::Allocated
        ||  state == LeafState::Wanted
        ||  state == LeafState::WantedHead
    }

    fn try_merge_buddies(&mut self, floating_id: usize) -> Option<usize> {
        assert!(floating_id < self.leaves.len());
        assert!(self.leaves[floating_id].is_off_lists());

        let result = self.buddy_id(floating_id, BuddyAction::Merge);

        result?;

        let buddy_id = result.unwrap();
        let buddy    = &self.leaves[buddy_id];
        let list_id  = self.leaves[floating_id].current_index;
        let state    = buddy.state;

        // If the buddy isn't free, we can't merge it.

        if !self.is_mergeable(state) {
            return None;
        }

        // If the buddy current is split to a smaller size, it can't be merged.

        if buddy.current_index != list_id {
            return None;
        }

        assert!((list_id as usize) < self.max_list_index() as usize);
        assert!(buddy.is_on_list(buddy.current_index as u32, self.leaves.len()));

        self.remove_free(list_id as usize, buddy_id).unwrap();

        // Get the id for the merged result.

        let merged_id = std::cmp::min(floating_id, buddy_id);

        // Clear the leaf state to merged the higher-addressed leaf.

        if merged_id == floating_id {
            self.leaves[buddy_id   ].state         = LeafState::Merged;
            self.leaves[buddy_id   ].current_index = 0;
        } else {
            self.leaves[floating_id].state         = LeafState::Merged;
            self.leaves[floating_id].current_index = 0;
        }

        // Now update the base leaf.

        self.leaves[merged_id  ].state         = LeafState::Free;
        self.leaves[merged_id  ].current_index = list_id + 1;

        self.merges += 1;

        Some(merged_id)
    }

    fn buddy_id(&self, leaf_id: usize, action: BuddyAction) -> Option<usize> {
        let     list_id = self.leaves[leaf_id].current_index;

        // If we are trying to merge and the leaf is already at the
        // maximum size, there's not buddy.

        if action == BuddyAction::Merge && list_id == self.max_list_index() {
            return None;
        }

        // If we are trying to split, and the buddy is already at the smallest
        // size, there's no buddy.

        if action == BuddyAction::Split && list_id == 0 {
            return None;
        }

        // Determine which buddy we want:  if we are merging,
        // then the buddy for the current size is correct.
        // If we are splitting, then the buddy wanted is for
        // the buddy for half the current size.

        let current_size = self.leaves[leaf_id].current_size(self.log2_min_size);

        let buddy_size =
            if action == BuddyAction::Merge {
                current_size
            } else {
                current_size / 2
            };

        // Now get the offset of the buddy, that is, the distance
        // to the buddy in the self.leaves array.

        let offset = buddy_size / self.min_size;

        assert!(offset.is_power_of_two());

        let buddy_id = leaf_id ^ offset;

        if buddy_id >= self.leaves.len() {
            return None;
        }

        Some(buddy_id)
    }

    fn max_list_index(&self) -> Index {
        let result = self.free_lists.len() - 1;

        result as Index
    }

    /// Returns the statistics for the pool.

    pub fn get_stats(&self) -> BuddyStats {
        let min_size     = self.min_size;
        let max_size     = self.max_size;
        let allocs       = self.allocs;
        let frees        = self.frees;
        let out_of_mem   = self.out_of_mem;
        let freeing_free = self.out_of_mem;
        let splits       = self.splits;
        let merges       = self.merges;

        let mut freelist_stats = Vec::with_capacity(self.free_lists.len());

        for freelist in &self.free_lists {
            let stats =
                ListStats {
                    size:      freelist.size,
                    enqueues:  freelist.enqueues,
                    dequeues:  freelist.dequeues,
                    removes:   freelist.removes,
                };

            freelist_stats.push(stats);
        }

        BuddyStats {
            min_size, max_size, allocs, frees, out_of_mem, freeing_free, splits, merges, freelist_stats
        }
    }

    // This helper function converts a byte count into a valid index into
    // the free lists, if possible, or returns an error, if not.

    fn compute_list_id(&self, bytes: usize) -> Result<usize, BuddyError> {
        if bytes > self.max_size {
            let message = format!("That size ({}) is too large.", bytes);

            return Err(BuddyError::new(BuddyErrorType::OversizeAlloc, message));
        }

        let bytes  = bytes.next_power_of_two();
        let bytes  = std::cmp::max(bytes, self.min_size);
        let index  = bytes.ilog2() as usize - self.log2_min_size;

        assert!(index < self.free_lists.len());
        Ok(index)
    }

    #[cfg(test)]
    fn freelist_buddy_size(&self, index: usize) -> usize {
        let shift = index + self.log2_min_size;

        2_usize.pow(shift as u32)
    }

    // Pushes a leaf onto the given free list.

    fn enqueue_free(&mut self, freelist_index: usize, leaf_index: usize) -> Result<(), BuddyError> {
        if freelist_index >= self.free_lists.len() {
            let message = format!("enqueue_free:  That free list ({}) is out of range.", freelist_index);

            return Err(BuddyError::new(BuddyErrorType::InvalidListId, message));
        }

        // Put the leaf on the list.

        Self::enqueue(&mut self.free_lists[freelist_index], &mut self.leaves, leaf_index)?;

        self.leaves[leaf_index].current_index = freelist_index as Index;
        self.leaves[leaf_index].state         = LeafState::Free;

        Ok(())
    }

    // Dequeues the head of the given free list.

    fn dequeue_free(&mut self, freelist_index: usize) -> Result<Option<usize>, BuddyError> {
        if freelist_index >= self.free_lists.len() {
            let message = format!("dequeue_free:  That free list ({}) is out of range.", freelist_index);

            return Err(BuddyError::new(BuddyErrorType::InvalidListId, message));
        }

        Ok(Self::dequeue(&mut self.free_lists[freelist_index], &mut self.leaves))
    }

    // Removes the specified leaf from an arbitrary point in the free list.

    fn remove_free(&mut self, freelist_index: usize, leaf_index: usize) -> Result<(), BuddyError> {
        if freelist_index >= self.free_lists.len() {
            let message = format!("remove_free:  That free list ({}) is out of range.", freelist_index);

            return Err(BuddyError::new(BuddyErrorType::InvalidListId, message));
        }

        Self::remove(&mut self.free_lists[freelist_index], &mut self.leaves, leaf_index)
    }

    // Puts a leaf at the head of a list.

    fn enqueue(head: &mut ListHead, leaves: &mut [Leaf], leaf_index: usize) -> Result<(), BuddyError> {
        if leaf_index >= leaves.len() {
            let message = format!("That leaf ({}) is out of range.", leaf_index);

            return Err(BuddyError::new(BuddyErrorType::InvalidLeafId, message));
        }

        if !leaves[leaf_index].is_off_lists() {
            let message = format!("That leaf ({}) already is on a list.", leaf_index);

            return Err(BuddyError::new(BuddyErrorType::AttachedLeaf, message));
        }

        if head.size == 0 {
            leaves[leaf_index].list_id = head.id;
            leaves[leaf_index].next    = leaf_index as Index;
            leaves[leaf_index].prev    = leaf_index as Index;

            head.first = leaf_index;
        } else {
            // Get the links for the new head.

            let new_next = head.first;
            let new_prev = leaves[head.first].prev as usize;

            // Initialize the new head of the list.

            leaves[leaf_index].list_id = head.id;
            leaves[leaf_index].next    = new_next as Index;
            leaves[leaf_index].prev    = new_prev as Index;

            // Point the new head's neighbors at the new head.

            leaves[new_next].prev = leaf_index as Index;
            leaves[new_prev].next = leaf_index as Index;

            // Set the head of the list.

            head.first = leaf_index;
        }

        head.size     += 1;
        head.enqueues += 1;
        Ok(())
    }

    // Dequeues the head of a list, if any, returning the index of the leaf
    // if the list is not empty.

    fn dequeue(head: &mut ListHead, leaves: &mut [Leaf]) -> Option<usize> {
        if head.size == 0 {
            None
        } else {
            let result = head.first;
            assert!(leaves[result].is_on_list(head.id, leaves.len()));

            if head.size == 1 {
                head.first = usize::MAX;
            } else {
                // Get the pointers.

                let free_leaf = result;
                let next_leaf = leaves[free_leaf].next as usize;
                let prev_leaf = leaves[free_leaf].prev as usize;

                // Update the pointers in the prev and next leaves.

                leaves[prev_leaf].next = next_leaf as Index;
                leaves[next_leaf].prev = prev_leaf as Index;

                // Set the new list head.

                head.first = next_leaf;
            }

            // Remove the list and pointers to other leaves in this list.

            leaves[result].invalidate_list_info();

            // Decrement the list size and keep the count of dequeue operations.

            head.size     -= 1;
            head.dequeues += 1;
            Some(result)
        }
    }

    // Removes an arbitrary element from a list.  This routine checks that the
    // leaf is on the given list.

    fn remove(head: &mut ListHead, leaves: &mut [Leaf], leaf_index: usize) -> Result<(), BuddyError> {
        if head.size == 0 {
            let message =
                format!
                (
                    "That list ({}) is empty",
                    head.id
                );

            return Err(BuddyError::new(BuddyErrorType::EmptyList, message));
        }

        if leaf_index >= leaves.len() {
            let message =
                format!
                (
                    "That leaf index ({}) is out of range ({}).",
                    leaf_index,
                    leaves.len()
                );

            return Err(BuddyError::new(BuddyErrorType::InvalidLeafId, message));
        }

        if leaves[leaf_index].list_id == u32::MAX {
            let message = format!("That leaf ({}) is not on a list", leaf_index);

            return Err(BuddyError::new(BuddyErrorType::UnattachedLeaf, message));
        }

        if leaves[leaf_index].list_id != head.id {
            let message =
                format!
                (
                    "That leaf ({} on list {}) is not on that list ({}).",
                    leaf_index,
                    leaves[leaf_index].list_id,
                    head.id
                );

            return Err(BuddyError::new(BuddyErrorType::ListIdMismatch, message));
        }

        // Set a new head as needed.

        if head.size == 1 {
            assert!(head.first == leaf_index);
            head.first = usize::MAX
        } else {
            // Set a new head if needed.

            if head.first == leaf_index {
                head.first = leaves[leaf_index].next as usize;
            }

            // Now unlink the element to be removed.

            let prev = leaves[leaf_index].prev as usize;
            let next = leaves[leaf_index].next as usize;

            leaves[prev].next = next as Index;
            leaves[next].prev = prev as Index;
        }

        // Mark the removed element as being off any list.

        leaves[leaf_index].invalidate_list_info();
        head.size    -= 1;
        head.removes += 1;
        Ok(())
    }

    // Converts a leaf into an address in the pool area.

    fn to_address(&self, leaf_index: usize) -> Result<usize, BuddyError> {
        if leaf_index >= self.leaves.len() {
            let message =
                format!
                (
                    "That leaf ({}) is out of range ({})",
                    leaf_index,
                    self.leaves.len()
                );

            return Err(BuddyError::new(BuddyErrorType::InvalidLeafId, message));
        }

        Ok(self.base + leaf_index * self.min_size)
    }
}

/// Converts a byte count into a buddy size, if possibe, obeying the
/// limits on buddy sizes given as parameters.

pub fn compute_buddy_size(size: usize, min_buddy: usize, max_buddy: usize) -> Result<usize, BuddyError> {
    if !min_buddy.is_power_of_two() {
        let message = format!( "That min_buddy ({}) is not a power of 2", min_buddy);
        return Err(BuddyError::new(BuddyErrorType::InvalidParameter, message));
    }

    if !max_buddy.is_power_of_two() {
        let message = format!( "That max_buddy ({}) is not a power of 2", max_buddy);
        return Err(BuddyError::new(BuddyErrorType::InvalidParameter, message));
    }

    if min_buddy > max_buddy {
        let message =
            format!
            (
                "That min_buddy ({}) is larger than the max_buddy ({}).",
                min_buddy,
                max_buddy
            );

        return Err(BuddyError::new(BuddyErrorType::InvalidParameter, message));
    }

    if size > max_buddy {
        let message = format!( "That size ({}) is too large ({})", size, max_buddy);
        return Err(BuddyError::new(BuddyErrorType::InvalidSize, message));
    }

    let buddy_size =
        if size < min_buddy {
            min_buddy
        } else {
            size.next_power_of_two()
        };

    Ok(buddy_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Check that an error result type matches expectations.

    macro_rules! check_error {
            ($x:expr, $y:ident) => { verify_error($x.unwrap_err(), BuddyErrorType::$y) }
        }

    // Check that an error returned the expected type and error message.

    macro_rules! check_error_message {
            ($x:expr, $y:ident, $z:expr) =>
                { verify_error_message($x.unwrap_err(), BuddyErrorType::$y, $z) }
        }

    fn verify_error(result: BuddyError, error_type: BuddyErrorType) {
        if result.error_type() != error_type {
            println!("verify_error:  {}", result.error_type());
        }

        assert!(result.error_type() == error_type);
    }

    fn verify_error_message(result: BuddyError, error_type: BuddyErrorType, message: &str) {
        verify_error(result.clone(), error_type);

        if result.message() != message {
            println!
            (
                "verify_error_message:\n    got      \"{}\"\n    expected \"{}\"",
                result.message(),
                message
            );
        }

        assert!(result.message() == message);
    }

    #[test]
    #[should_panic]
    fn test_verify_error_no_error() {
        let result = Ok(());

        check_error!(result, InvalidListId);
    }

    #[test]
    #[should_panic]
    fn test_verify_error_type_mismatch() {
        let result: Result<(), BuddyError> = Err(BuddyError::new_str(BuddyErrorType::InvalidListId, &"test"));

        check_error!(result, InvalidParameter);
    }

    #[test]
    fn test_verify_error_pass() {
        let result: Result<(), BuddyError> = Err(BuddyError::new_str(BuddyErrorType::InvalidParameter, &"test"));

        check_error!(result, InvalidParameter);
    }

    #[test]
    #[should_panic]
    fn test_verify_error_message_type_mismatch() {
        let result: Result<(), BuddyError> = Err(BuddyError::new_str(BuddyErrorType::InvalidListId, &"test"));

        check_error_message!(result, InvalidParameter, "test");
    }

    #[test]
    fn test_verify_error_message_pass() {
        let result: Result<(), BuddyError> = Err(BuddyError::new_str(BuddyErrorType::InvalidListId, &"test"));

        check_error_message!(result, InvalidListId, "test");
    }

    #[test]
    #[should_panic]
    fn test_verify_error_message_message_mismatch() {
        let result: Result<(), BuddyError> = Err(BuddyError::new_str(BuddyErrorType::InvalidListId, &"test"));

        check_error_message!(result, InvalidListId, "mismatch");
    }

    #[test]
    fn test_enqueue_dequeue_type_mismatch() {
        let     count     = 100;
        let mut head      = ListHead::new(1);
        let mut leaves    = Vec::with_capacity(count);

        for i in 0..count {
            let leaf = Leaf::new(LeafState::Free, 0);

            leaves.push(leaf);

            assert!(leaves[i].is_off_lists());
            assert!(leaves[i].is_list_consistent(0, leaves.len()));
        }

        // Test enqueueing an invalid leaf id.

        let result = BuddyPool::enqueue(&mut head, &mut leaves, count);

        check_error!(result, InvalidLeafId);

        // Build the list.

        for i in 0..leaves.len() {
            BuddyPool::enqueue(&mut head, &mut leaves, i).unwrap();

            assert!(head.size  == i + 1);
            assert!(head.first == i    );

            assert!(leaves[i].is_on_list(head.id, leaves.len()));

            // Try enqueueing the leaf again...

            let result = BuddyPool::enqueue(&mut head, &mut leaves, i);

            check_error!(result, AttachedLeaf);
        }

        for i in 0..leaves.len() {
            assert!(leaves[head.first].is_on_list(head.id, leaves.len()));

            let leaf_index = BuddyPool::dequeue(&mut head, &mut leaves).unwrap();

            assert!(leaves[leaf_index].is_off_lists());
            assert!(leaves[leaf_index].is_list_consistent(0, leaves.len()));

            assert!(leaf_index == leaves.len() - 1 - i);
            assert!(head.size  == leaves.len() - 1 - i);
        }

        assert!(head.size  == 0);
        assert!(head.first == usize::MAX);

        assert!(BuddyPool::dequeue(&mut head, &mut leaves).is_none());

        for i in 0..count / 2 {
            BuddyPool::enqueue(&mut head, &mut leaves, i).unwrap();
            assert!(leaves[head.first].is_on_list(head.id, leaves.len()));
            assert!(head.first == i);

            BuddyPool::enqueue(&mut head, &mut leaves, i + count / 2).unwrap();

            assert!(leaves[head.first].is_on_list(head.id, leaves.len()));
            assert!(head.first == i + count / 2);
            assert!(head.size  == (i + 1) * 2);
        }

        for i in 0..count / 2 {
            assert!(head.first == count - 1 - i);
            let leaf_index = BuddyPool::dequeue(&mut head, &mut leaves).unwrap();
            assert!(leaf_index == count - 1 - i);

            assert!(head.first == count / 2 - 1 - i);
            let leaf_index = BuddyPool::dequeue(&mut head, &mut leaves).unwrap();
            assert!(leaf_index == count / 2 - 1 - i);
        }

        assert!(BuddyPool::dequeue(&mut head, &mut leaves).is_none());
    }

    #[test]
    fn test_remove() {
        let     count     = 100;
        let mut head      = ListHead::new(1);
        let mut leaves    = Vec::with_capacity(count);
        let mut removed   = ListHead::new(2);

        for i in 0..count {
            let leaf = Leaf::new(LeafState::Free, 0);

            leaves.push(leaf);

            assert!(leaves[i].is_off_lists());
            assert!(leaves[i].is_list_consistent(0, leaves.len()));
        }

        for i in 0..leaves.len() {
            BuddyPool::enqueue(&mut head, &mut leaves, i).unwrap();

            assert!(head.size  == i + 1);
            assert!(head.first == i    );

            assert!(leaves[i].is_on_list(head.id, leaves.len()));
        }

        // Now check an invalid leaf index.

        let leaf_index  = leaves.len();
        let result      = BuddyPool::remove(&mut head, &mut leaves, leaf_index);

        check_error!(result, InvalidLeafId);

        // Try removing the head of the list.

        let leaf_index = head.first;

        let result = BuddyPool::remove(&mut head, &mut leaves, leaf_index);

        assert!(result.is_ok());
        assert!(leaves[leaf_index].is_off_lists());
        assert!(leaves[leaf_index].is_list_consistent(0, leaves.len()));
        assert!(head.size  == leaves.len() - 1);
        assert!(head.first == leaf_index - 1);

        // Okay, put it back on the list.

        BuddyPool::enqueue(&mut head, &mut leaves, leaf_index).unwrap();

        for i in 0..count {
            // Remove a leaf on the list.  This operation should succeed.

            assert!(leaves[i].is_on_list(head.id, leaves.len()));
            let result = BuddyPool::remove(&mut head, &mut leaves, i);

            assert!(result.is_ok());
            assert!(leaves[i].is_off_lists());
            assert!(leaves[i].is_list_consistent(0, leaves.len()));
            assert!(head.size == leaves.len() - 1 - i);

            // Try to remove the leaf again.  This operation should fail.

            let result = BuddyPool::remove(&mut head, &mut leaves, i);

            assert!(result.is_err());

            // Check that we get the correct error type.

            if head.size > 0 {
                check_error!(result, UnattachedLeaf);

                // Now try putting the leaf on a different list, and then remove
                // it.  We should get a list mismatch.

                BuddyPool::enqueue(&mut removed, &mut leaves, i).unwrap();

                let result = BuddyPool::remove(&mut head, &mut leaves, i);

                check_error!(result, ListIdMismatch);

                // Okay, remove this leaf from the "removed" list.

                let result = BuddyPool::dequeue(&mut removed, &mut leaves).unwrap();

                assert!(result == i);
            } else {
                check_error!(result, EmptyList);
            }

            assert!(head.size  == leaves.len() - 1 - i);
        }

        assert!(head.size  == 0);
        assert!(head.first == usize::MAX);
    }

    fn test_limits(min_buddy: usize, max_buddy: usize) {
        let mut start          = 0;
        let mut current_buddy  = min_buddy;

        while current_buddy <= max_buddy {
            for i in start..current_buddy {
                let buddy_size = compute_buddy_size(i, min_buddy, max_buddy).unwrap();

                assert!(buddy_size == current_buddy);
            }

            start          = current_buddy + 1;
            current_buddy *= 2;
        }

        let result = compute_buddy_size(max_buddy, min_buddy, max_buddy).unwrap();
        assert!(result == max_buddy);

        let result = compute_buddy_size(max_buddy + 1, min_buddy, max_buddy);

        check_error!(result, InvalidSize);
    }

    #[test]
    fn test_compute_buddy_size() {
        let min_buddy = 1024;
        let max_buddy = 16 * 1024 * 1024;

        test_limits(min_buddy, max_buddy);

        // Test an invalid min_buddy.

        let result = compute_buddy_size(1, 1023, 2048);

        check_error!(result, InvalidParameter);

        // Test a min_buddy of zero.

        let result = compute_buddy_size(1, 0, 2048);

        check_error!(result, InvalidParameter);

        // Test an invalid max_buddy.

        let result = compute_buddy_size(1, 1024, 2049);

        check_error!(result, InvalidParameter);

        // Test a max_buddy of zero.

        let result = compute_buddy_size(1, 1024, 0);

        check_error!(result, InvalidParameter);

        // Test a min_buddy that is larger than max_buddy.

        let result = compute_buddy_size(1, 1024, 512);

        check_error!(result, InvalidParameter);

        let result = compute_buddy_size(8192, 1024, 4096);

        check_error!(result, InvalidSize);
    }

    fn get_simple_pool() -> BuddyPool {
        let  base       = 0x10000000;
        let  min_alloc  = 0;
        let  min_buddy  = 1024;
        let  max_alloc  = 8192;

        // For small pools, allocate as many leaves as possible, to
        // test corner cases that have failed in the past.

        let leaves =
            if Index::MAX as usize >= u32::MAX as usize {
                base / min_buddy
            } else {
                Index::MAX as usize + 1
            };

        let  size   = std::cmp::min(base / 2, leaves * min_buddy);
        let  config = BuddyConfig { base, size, min_alloc, min_buddy, max_alloc };

        BuddyPool::new(config).unwrap()
    }

    fn get_non_empty_freelist(pool: &BuddyPool) -> Option<usize> {
        for i in 0..pool.free_lists.len() {
            if pool.free_lists[i].size > 0 {
                return Some(i);
            }
        }

        None
    }

    #[test]
    fn test_get_non_empty_freelist() {
        let mut pool = get_simple_pool();

        for i in 0..pool.free_lists.len() {
            loop {
                let free_leaf = pool.dequeue_free(i).unwrap();

                if free_leaf.is_none() {
                    break;
                }
            }
        }

        assert!(get_non_empty_freelist(&pool).is_none());
    }

    #[test]
    fn test_freelist() {
        let mut pool  = get_simple_pool();
        let     size  = pool.end - pool.base;

        assert!(pool.min_size == 1024);

        let mut non_empty  = usize::MAX;
        let     free_count = size / pool.max_size;
        let     free_index = pool.max_size.ilog2() as usize - pool.log2_min_size;

        for i in 0..pool.free_lists.len() {
            if i != free_index {
                assert!(pool.free_lists[i].size == 0);
            } else {
                assert!(pool.free_lists[i].size == free_count);
                non_empty = i;
            }
        }

        assert!(non_empty < pool.free_lists.len());
        let merge_count = 2_usize.pow(non_empty as u32);

        for i in 0..pool.leaves.len() {
            if i % merge_count == 0 {
                assert!(pool.leaves[i].is_on_list(pool.free_lists[non_empty].id, pool.leaves.len()));
                assert!(pool.leaves[i].state == LeafState::Free);
                assert!(pool.leaves[i].current_index == pool.max_list_index());
            } else {
                assert!(pool.leaves[i].is_off_lists());
                assert!(pool.leaves[i].is_list_consistent(pool.free_lists.len() as u32, pool.leaves.len()));
                assert!(pool.leaves[i].state == LeafState::Merged);
                assert!(pool.leaves[i].current_index == 0);
            }

            assert!(pool.to_address(i).unwrap() == pool.base + i * pool.min_size);
        }

        // Give to_address an invalid leaf id.

        let result = pool.to_address(pool.leaves.len());

        check_error!(result, InvalidLeafId);

        // Try dequeueing from an invalid free list.

        let result = pool.dequeue_free(pool.free_lists.len());

        check_error!(result, InvalidListId);

        // Try dequeueing some leaves.

        let leaf_index_1 = pool.dequeue_free(non_empty).unwrap().unwrap();

        assert!(leaf_index_1 == pool.leaves.len() - merge_count);
        assert!(pool.free_lists[non_empty].size == free_count - 1);
        assert!(pool.to_address(leaf_index_1).unwrap() == pool.base + leaf_index_1 * pool.min_size);

        let leaf_index_2 = pool.dequeue_free(non_empty).unwrap().unwrap();
        assert!(leaf_index_2 == pool.leaves.len() - 2 * merge_count);
        assert!(pool.free_lists[non_empty].size == free_count - 2);

        // Try an invalid list for enqueue.

        let result = pool.enqueue_free(pool.free_lists.len(), pool.leaves.len() - merge_count);

        check_error!(result, InvalidListId);

        // Try an invalid leaf id.

        let result = pool.enqueue_free(0, pool.leaves.len());

        check_error!(result, InvalidLeafId);

        // Okay, enqueue the leaves back onto the free list.

        pool.leaves[leaf_index_1].state = LeafState::Allocated;
        pool.leaves[leaf_index_2].state = LeafState::Allocated;

        pool.enqueue_free(non_empty, leaf_index_1).unwrap();
        pool.enqueue_free(non_empty, leaf_index_2).unwrap();

        assert!(pool.leaves[leaf_index_1].state == LeafState::Free);
        assert!(pool.leaves[leaf_index_2].state == LeafState::Free);
        assert!(pool.free_lists[non_empty].size == free_count);

        // Check removing a leaf not on the list.

        pool.remove_free(non_empty, pool.leaves.len() - merge_count).unwrap();

        let result = pool.remove_free(non_empty, pool.leaves.len() - merge_count);

        check_error!(result, UnattachedLeaf);

        // Try remove on an invalid leaf id.

        let result = pool.remove_free(non_empty, pool.leaves.len());

        check_error!(result, InvalidLeafId);

        // Try remove on an invalid list id.

        let result = pool.remove_free(pool.free_lists.len(), pool.leaves.len() - 4 * merge_count);

        check_error!(result, InvalidListId);
    }

    #[test]
    fn test_compute_list_id() {
        let     pool          = get_simple_pool();
        let mut current_size  = pool.min_size;
        let mut current_index = 0;

        for i in 0..pool.max_size {
            if i > current_size {
                current_index += 1;
                current_size  *= 2;
            }

            let log_index = pool.compute_list_id(i).unwrap();

            assert!(log_index == current_index);
        }

        let result = pool.compute_list_id(pool.max_size + 1);

        check_error!(result, OversizeAlloc);
    }

    #[test]
    fn test_get_stats() {
        let mut pool  = get_simple_pool();
        let mut index = pool.free_lists.len();

        // Find the free list that contains all the free blocks.

        for i in 0..pool.free_lists.len() {
            if pool.free_lists[i].size != 0 {
                let list   = &mut pool.free_lists[i];
                let expect = list.first;
                let result = BuddyPool::dequeue(list, &mut pool.leaves).unwrap();

                // Get the limits for some entries in the leaf structure.

                let leaf_limit     = pool.leaves.len();
                let freelist_limit = pool.free_lists.len() as u32;

                assert!(result == expect);
                assert!(pool.leaves[result].is_off_lists());
                assert!(pool.leaves[result].is_list_consistent(freelist_limit, leaf_limit));

                index = i;
                break;
            }
        }

        // Make sure we found a list with members.

        assert!(index < pool.free_lists.len());

        // Now check the enqueue and dequeue counts.

        let size     = pool.end - pool.base;
        let enqueues = size / pool.max_size;
        let stats    = pool.get_stats();

        assert!(stats.allocs == 0);
        assert!(stats.frees  == 0);
        assert!(stats.splits == 0);
        assert!(stats.merges == 0);

        // Check that our dequeue was counted, and the enqueues were
        // as well.  The enqueues occurred when we set up the leaves
        // array and the free lists.

        assert!(stats.freelist_stats[index].dequeues == 1       );
        assert!(stats.freelist_stats[index].enqueues == enqueues);

        for i in 0..pool.free_lists.len() {
            if i != index {
                assert!(stats.freelist_stats[i].dequeues == 0);
                assert!(stats.freelist_stats[i].enqueues == 0);
            }
        }
    }

    #[test]
    fn test_setup_leaves() {
        let mut leaves     = Vec::new();
        let mut freelists  = Vec::new();

        freelists.push(ListHead::new(0));

        let result = BuddyPool::setup_leaves(&mut leaves, &mut freelists, 1024, 8192);

        check_error!(result, EmptyPool);
    }

    #[test]
    fn test_display() {
        assert!(format!("{}", LeafState::Free      ) == "Free"      );
        assert!(format!("{}", LeafState::Allocated ) == "Allocated" );
        assert!(format!("{}", LeafState::Merged    ) == "Merged"    );
        assert!(format!("{}", LeafState::Wanted    ) == "Wanted"    );
        assert!(format!("{}", LeafState::WantedHead) == "WantedHead");
        assert!(format!("{}", LeafState::Locked    ) == "Locked"    );
        assert!(format!("{}", LeafState::LockedHead) == "LockedHead");

        assert!(format!("{}", BuddyErrorType::InvalidParameter) == "InvalidParameter");
    }

    fn test_bad_config(config: BuddyConfig, message: &str) {
        let result = BuddyPool::new(config);

        check_error_message!(result, InvalidParameter, message);
    }

    fn test_pool_size(mut config: BuddyConfig, size: usize) {
        config.size = size;

        let message =
            format!
            (
                "The pool size ({}) is out of the range [1, usize::MAX / 2]",
                config.size
            );

        test_bad_config(config, &message);
    }

    #[test]
    fn test_bad_pool_config() {
        let base       = 0x10_000;
        let size       = base / 2;
        let min_alloc  = 0;
        let min_buddy  = 1024;
        let max_alloc  = size / 4;

        let base_config =
            BuddyConfig {
                base,
                size,
                min_alloc,
                min_buddy,
                max_alloc,
            };

        let mut config = base_config.clone();

        assert!(BuddyPool::new(config).is_ok());

        // Now test the pool size checks.

        config           = base_config.clone();
        config.min_buddy = 512;
        config.max_alloc = 1011;

        test_pool_size(config, 0);

        config           = base_config.clone();
        config.min_buddy = 512;
        config.max_alloc = 1011;
        config.size      = BuddyPool::truncate(usize::MAX / 2 + 4 * base_config.min_buddy, base_config.min_buddy);

        test_pool_size(config, config.size);

        // Check some of the max_alloc verification.

        config           = base_config.clone();
        config.max_alloc = config.size * 2;

        let message =
            format!
            (
                "The max allocation size ({}) is larger than the pool size ({}).",
                config.max_alloc,
                config.size
            );

        test_bad_config(config, &message);

        // Test a very large max_alloc.

        config.max_alloc = usize::MAX;

        let message =
            "The max_alloc value (18446744073709551615) is out of the range [1, usize::MAX / 2]";

        test_bad_config(config, message);

        // max_alloc of zero is invalid, too.

        config.max_alloc = 0;

        let message =
            "The max_alloc value (0) is out of the range [1, usize::MAX / 2]";

        test_bad_config(config, &message);

        // The min_alloc must be <= max_alloc

        config           = base_config.clone();
        config.max_alloc = 8192;
        config.min_alloc = config.max_alloc * 2;

        let message =
            "The min allocation size (16384) is larger than the max allocation size (8192).";

        test_bad_config(config, &message);

        // Now check the max_alloc size after rounding up to its buddy size.

        config.min_alloc = 1024;
        config.size      = 8192 + config.min_alloc;
        config.max_alloc = 8193;

        let message =
            "The buddy size (16384) of the max allocation size (8193) exceeds the pool size (9216).";

        test_bad_config(config, &message);

        config.size      = base / 4;
        config.min_alloc = 1024;
        config.max_alloc = 8192;
        config.min_buddy = 16384;

        let message =
            "The min buddy size (16384) is larger than the max buddy size (8192).";

        test_bad_config(config, &message);

        config           = config.clone();
        config.min_alloc = usize::MAX;

        let message =
            "The min_alloc value (18446744073709551615) is out of the range [0, usize::MAX / 2]";

        test_bad_config(config, &message);

        // Check a zero min_buddy...

        config           = base_config.clone();
        config.min_buddy = 0;

        let message =
            format!("The min_buddy value ({}) is out of the range [1, usize::MAX / 2]", config.min_buddy);

        test_bad_config(config, &message);

        // Check a large min_buddy...

        config           = base_config.clone();
        config.min_buddy = usize::MAX;

        let message =
            format!("The min_buddy value ({}) is out of the range [1, usize::MAX / 2]", config.min_buddy);

        test_bad_config(config, &message);

        // Now check an unfriendly base parameter.

        config      = base_config.clone();
        config.base = usize::MAX;

        let max_base = (usize::MAX / 2) - config.size - config.min_buddy;

        let message = format!
        (
            "The base ({}) is out of the range [0, {}]",
            config.base,
            max_base
        );

        test_bad_config(config, &message);
    }

    #[test]
    fn test_is_list_consistent() {
        let     leaf_count = 4;
        let mut leaf       = Leaf::new(LeafState::Free, 4);

        assert!(leaf.is_list_consistent(0, leaf_count));

        // Check the handling of leaf.next.

        leaf.next = leaf_count as Index;
        assert!(!leaf.is_list_consistent(0, leaf_count));

        leaf.next = 0;
        assert!(!leaf.is_list_consistent(0, leaf_count));

        leaf.next = Index::MAX;
        assert!(leaf.is_list_consistent(0, leaf_count));

        // Now check leaf.prev.

        leaf.prev = leaf_count as Index;
        assert!(!leaf.is_list_consistent(0, leaf_count));

        leaf.prev = 0;
        assert!(!leaf.is_list_consistent(0, leaf_count));

        leaf.prev = Index::MAX;
        assert!(leaf.is_list_consistent(0, leaf_count));

        // Now check the list_id handling.

        leaf.list_id = 1;
        assert!(!leaf.is_list_consistent(2, leaf_count));

        // Now try making a leaf look like it's on a list.

        leaf.next = 0;
        leaf.prev = 1;

        assert!( leaf.is_list_consistent(2, leaf_count));

        // Check that the list_id limit is checked properly.

        assert!(!leaf.is_list_consistent(1, leaf_count));
    }

    #[test]
    fn test_alloc_all() {
        let mut pool   = get_simple_pool();

        // Find a free list with some entries.

        let index = get_non_empty_freelist(&pool).unwrap();

        // The other lists should be empty.

        for i in 0..pool.free_lists.len() {
            if i != index {
                assert!(pool.free_lists[i].size == 0);
            }
        }

        // Compute the size of entries in this free list.

        let     shift       = index + pool.log2_min_size;
        let     alloc_size  = 2_usize.pow(shift as u32);
        let mut allocs      = 0;

        for _i in 0..pool.free_lists[index].size {
            let address = pool.alloc(alloc_size).unwrap();

            assert!(address >= pool.base && address < pool.end);
            allocs += 1;
        }

        // Try one more alloc to get a failure.  Use a smaller
        // size to test the search for a free block a bit better.

        allocs += 1;

        let last_alloc = alloc_size / 4;
        let result     = pool.alloc(last_alloc);

        let message =
            format!
            (
                "The pool has no memory available for that request size ({})",
                last_alloc
            );

        check_error_message!(result, OutOfMemory, &message);

        let stats = pool.get_stats();

        assert!(stats.out_of_mem == 1     );
        assert!(stats.allocs     == allocs);
    }

    #[test]
    fn test_simple_alloc() {
        let mut pool       = get_simple_pool();
        let mut index      = pool.free_lists.len();
        let     pool_size  = pool.end - pool.base;
        let     blocks     = pool_size / pool.max_alloc;

        for i in 0..pool.free_lists.len() {
            if pool.free_lists[i].size != 0 {
                index = i;
                break;
            }
        }

        assert!(index < pool.free_lists.len());
        assert!(pool.free_lists[index].size == blocks);

        let shift      = index + pool.log2_min_size;
        let alloc_size = 2_usize.pow(shift as u32);

        // Try allocating one block.  There should be no split.

        let address = pool.alloc(alloc_size).unwrap();
        assert!(address == pool.end - alloc_size);
        assert!(pool.free_lists[index].size == blocks - 1);

        // Now check an alloc attempt with an oversize request.

        let result = pool.alloc(pool.max_alloc + 1);
        check_error!(result, OversizeAlloc);

        // Now check an undersize request.

        pool.min_alloc = 1;

        let result = pool.alloc(0);
        check_error!(result, UndersizeAlloc);

        // Try passing an address before the pool base to free.

        let result  = pool.free(pool.base - 1);

        let message =
            format!
            (
                "That address ({:#x}) is out of range ([{:#x}, {:#x}])", pool.base - 1, pool.base, pool.end
            );

        check_error_message!(result, InvalidAddress, &message);

        // Try passing an address after the pool end to free.

        let result  = pool.free(pool.end + 1);
        let message =
            format!
            (
                "That address ({:#x}) is out of range ([{:#x}, {:#x}])", pool.end + 1, pool.base, pool.end
            );

        check_error_message!(result, InvalidAddress, &message);

        // Try passing a misaligned address.

        let result  = pool.free(address + 1);
        let message =
            format!
            (
                "That address ({:#x}) is incorrect (closest buddy {:#x})", address + 1, address
            );

        check_error_message!(result, InvalidAddress, &message);

        // Now check invalid states.

        let leaf_index = (address - pool.base) / pool.min_size;

        pool.leaves[leaf_index].state = LeafState::Free;

        let result = pool.free(address);

        let message =
            format!("That address ({:#x}) is not allocated (Free).", address);

        check_error_message!(result, FreeingFreeMemory, &message);

        // Free the memory after fixing the state.

        pool.leaves[leaf_index].state = LeafState::Allocated;

        pool.free(address).unwrap();

        // Now try a double free.

        let result  = pool.free(address);

        let message =
            format!
            (
                "That address ({:#x}) is not allocated ({}).", address, LeafState::Free
            );

        check_error_message!(result, FreeingFreeMemory, &message);
    }

    #[test]
    fn test_print_pool() {
        let pool = get_simple_pool();

        // Just make sure that they don't cause a fault.

        println!("test_print_pool:  pool:  {}", pool);
        println!("test_print_pool:  pool:  {:?}", pool);
    }

    #[test]
    fn test_index_limit() {
        if Index::MAX as usize <= u32::MAX as usize {
            let  base       = 0x10000000;
            let  min_alloc  = 0;
            let  min_buddy  = 1024;
            let  max_leaves = Index::MAX as usize + 1;
            let  size       = (max_leaves + 1) * min_buddy;
            let  max_alloc  = size / 8;
            let  config     = BuddyConfig { base, size, min_alloc, min_buddy, max_alloc };
            let  result     = BuddyPool::new(config);

            let message =
                format!
                (
                    "That Index type (max {}) is too small for the pool size ({}) with min_buddy {}.",
                    Index::MAX,
                    size,
                    min_buddy
                );

            check_error_message!(result, IndexTypeOverflow, &message);

            // If the index type is small enough, try creating a maximum-sized pool
            // for the given min_buddy.

            if Index::MAX as usize <= u16::MAX as usize {
                let mut config = config.clone();

                config.size -= config.min_buddy;

                let pool = BuddyPool::new(config).unwrap();

                assert!(pool.leaves.len() == max_leaves);
            }
        }
    }

    fn compare_message(got: &str, expected: &str) {
        if got != expected {
            println!
            (
                "compare_message:\n    got      \"{}\"\n    expected \"{}\"",
                got, expected
            );
        }

        assert!(got == expected);
    }

    #[test]
    #[should_panic]
    fn test_compare_message() {
        compare_message("a", "b");
    }

    #[test]
    fn test_dump_address() {
        let mut pool    = get_simple_pool();
        let     size    = pool.max_size;
        let     address = pool.alloc(size).unwrap();

        let message = pool.dump_address(address);

        compare_message
        (
            &message,
            &format!("{:#x} => state Allocated, size 8192, aligned = true", address)
        );

        let message = pool.dump_address(address + 1);

        compare_message
        (
            &message,
            &format!("{:#x} => state Allocated, size 8192, aligned = false", address + 1)
        );

        let message = pool.dump_address(address - 1);

        compare_message
        (
            &message,
            &format!("{:#x} => state Merged, size 1024, aligned = false", address - 1)
        );

        let message = pool.dump_address(pool.end);

        compare_message
        (
            &message,
            &format!("{:#x} => invalid address", pool.end)
        );
    }

    #[test]
    fn test_split_buddy() {
        let mut pool   = get_simple_pool();
        let mut index  = pool.free_lists.len();
        let mut larger = 0;

        // Start with the next to smallest buddy size so that
        // we can split our block.

        for i in 1..pool.free_lists.len() {
            larger = pool.free_lists[i].size;

            if larger > 0 {
                index = i;
                break;
            }
        }

        assert!(index < pool.free_lists.len());

        let leaf_id = pool.free_lists[index    ].first;
        let smaller = pool.free_lists[index - 1].size;

        assert!(pool.leaves[leaf_id].current_index == index as Index);

        // Get the block off the free list.

        pool.remove_free(index, leaf_id).unwrap();

        pool.split_buddy(leaf_id);

        assert!(pool.leaves[leaf_id].current_index == index as Index - 1);

        // Check that the free list lengths are correct.  Note that
        // split_buddy does not place the first half of the block
        // that was split onto the smaller list.  It is leaked in
        // this test.

        assert!(pool.free_lists[index    ].size == larger  - 1);
        assert!(pool.free_lists[index - 1].size == smaller + 1);

        // Get another pool for trying an allocate.

        let mut pool       = get_simple_pool();
        let     first      = pool.free_lists[index].first;
        let     address    = pool.to_address(first).unwrap();
        let     buddy_size = pool.leaves[first].current_size(pool.log2_min_size);
        let     alloc_size = buddy_size / 2;

        let result = pool.alloc(alloc_size).unwrap();

        assert!(address == result);
    }

    #[test]
    fn test_buddy_id() {
        let mut pool     = get_simple_pool();
        let     list_id  = get_non_empty_freelist(&pool).unwrap();

        assert!(pool.free_lists[list_id].size > 1);

        let float_id = pool.dequeue_free(list_id).unwrap().unwrap();

        assert!(pool.buddy_id(float_id, BuddyAction::Merge).is_none());
        assert!(pool.buddy_id(float_id, BuddyAction::Split).is_some());

        pool.split_buddy(float_id);

        let bytes    = pool.leaves[float_id].current_size(pool.log2_min_size);
        let skip     = bytes / pool.min_size;
        let buddy_id = float_id + skip;

        // Now that we have split them, see that we get reasonable buddy ids.
        // Since they are split, we need the ids for a merge.

        assert!(pool.buddy_id(buddy_id, BuddyAction::Merge).unwrap() == float_id);
        assert!(pool.buddy_id(float_id, BuddyAction::Merge).unwrap() == buddy_id);
    }

    #[test]
    fn test_merge_buddies() {
        let mut pool     = get_simple_pool();
        let     list_id  = get_non_empty_freelist(&pool).unwrap();
        let     shift    = list_id + pool.log2_min_size - 1;
        let     skip     = 2_usize.pow(shift as u32) / pool.min_size;

        // We need to be able to split the leaf.

        assert!(list_id > 0);

        let leaf_id = pool.dequeue_free(list_id).unwrap().unwrap();

        // The leaf should not be mergeable at this point.

        assert!(pool.buddy_id(leaf_id, BuddyAction::Merge).is_none());

        pool.split_buddy(leaf_id);

        // Okay, get the id now that they are split.  We thus need the
        // id for a merge.

        let buddy_id = pool.buddy_id(leaf_id, BuddyAction::Merge).unwrap();
        let list_id  = list_id - 1;

        assert!(buddy_id == leaf_id + skip);

        assert!(pool.leaves[buddy_id].is_on_list(list_id as u32, pool.leaves.len()));
        assert!(pool.leaves[buddy_id].state == LeafState::Free);

        let result = pool.try_merge_buddies(leaf_id).unwrap();

        assert!(result == leaf_id);
        assert!(pool.leaves[buddy_id].is_off_lists());
        assert!(pool.leaves[leaf_id ].is_off_lists());

        assert!(pool.leaves[leaf_id ].state         == LeafState::Free     );
        assert!(pool.leaves[leaf_id ].current_index == list_id as Index + 1);

        assert!(pool.leaves[buddy_id].state         == LeafState::Merged   );
    }

    #[test]
    fn test_one_buddy_pool() {
        let  base       = 0x10000000;
        let  min_alloc  = 0;
        let  min_buddy  = 1024;
        let  max_alloc  = 8192;

        // For small pools, allocate as many leaves as possible, to
        // test corner cases that have failed in the past.

        let leaves =
            if Index::MAX as usize >= u32::MAX as usize {
                base / min_buddy
            } else {
                Index::MAX as usize + 1
            };

        let  size   = std::cmp::min(base / 2, leaves * min_buddy);
        let  config = BuddyConfig { base, size, min_alloc, min_buddy, max_alloc };

        let mut pool       = BuddyPool::new(config).unwrap();
        let mut list_sizes = Vec::new();
        let mut addresses  = Vec::new();

        // Save the sizes of the free lists so that we can check for
        // correct merging on free.

        for list in &pool.free_lists {
            list_sizes.push(list.size);
        }

        for _ in 0..10 {
            let address = pool.alloc(max_alloc).unwrap();

            addresses.push(address);
        }

        for address in addresses {
            pool.free(address).unwrap();
        }

        check_pool_sizes(&pool, &list_sizes);

        // Now try operations that cause splits and merges.

        let mut addresses = Vec::new();

        for _ in 0..10 {
            let address = pool.alloc(min_alloc).unwrap();

            addresses.push(address);
        }

        for address in addresses {
            pool.free(address).unwrap();
        }

        check_pool_sizes(&pool, &list_sizes);

        // Make sure we cover the small sizes.

        assert!(pool.min_alloc == 0);

        for size in pool.min_alloc..=pool.max_alloc {
            let mut addresses = Vec::new();

            for _i in 0..10 {
                let address = pool.alloc(size).unwrap();

                addresses.push(address);
            }

            for address in addresses {
                pool.free(address).unwrap();
            }

            check_pool_sizes(&pool, &list_sizes);
        }

        assert!(verify_pool(&pool));

        check_pool_sizes(&pool, &list_sizes);
    }

    #[test]
    fn test_reverse() {
        let mut pool       = get_simple_pool();
        let mut list_sizes = Vec::new();
        let mut addresses  = Vec::new();

        for list in &pool.free_lists {
            list_sizes.push(list.size);
        }

        for _i in 0..10 {
            let mut size = pool.min_size;

            while size <= pool.max_size {
                let address = pool.alloc(size).unwrap();

                addresses.push(address);

                size *= 2;
            }
        }

        let mut index = addresses.len();

        while index > 0 {
            index -= 1;

            pool.free(addresses[index]).unwrap();
        }

        check_pool_sizes(&pool, &list_sizes);
        verify_pool(&pool);
    }

    fn check_pool_sizes(pool: &BuddyPool, list_sizes: &[usize]) {
        for i in 0..pool.free_lists.len() {
            if pool.free_lists[i].size != list_sizes[i] {
                println!
                (
                    "check_pool_sizes:  list size mismatch at free_lists[{}], got {}, expected {}",
                    i,
                    pool.free_lists[i].size,
                    list_sizes[i]
                );
            }

            assert!(pool.free_lists[i].size == list_sizes[i]);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_pool_sizes() {
        let mut pool  = get_simple_pool();
        let mut sizes = Vec::new();
        let     index = get_non_empty_freelist(&pool).unwrap();

        for list in &pool.free_lists {
            sizes.push(list.size);
        }

        assert!(verify_pool(&pool));

        pool.dequeue_free(index).unwrap().unwrap();

        check_pool_sizes(&pool, &sizes);
    }

    fn verify_pool(pool: &BuddyPool) -> bool {
        let mut total_mem = 0;

        for i in 0..pool.free_lists.len() {
            let valid = verify_list(&pool, &pool.free_lists[i], &format!("free_lists[{}]", i));

            if !valid {
                return false;
            }

            let size = pool.freelist_buddy_size(i);

            total_mem += size * pool.free_lists[i].size;
        }

        let pool_size = pool.end - pool.base;
        let pass      = total_mem == pool_size;

        if !pass {
            println!("verify_pool:  The size total is wrong");
            println!("    got      {}", total_mem);
            println!("    expected {}", pool_size);
        }

        pass
    }

    fn verify_list(pool: &BuddyPool, head: &ListHead, name: &str) -> bool {
        if head.size == 0 {
            if head.first != usize::MAX {
                println!("verify_list:  {}:  head of null list is {}", name, head.first);
            }

            return head.first == usize::MAX;
        }

        let mut current    = head.first as Index;
        let mut prev       = pool.leaves[head.first].prev;
        let     leaf_limit = pool.leaves.len();
        let     first      = head.first;

        assert!((current as usize) < leaf_limit);
        assert!((prev    as usize) < leaf_limit);

        // Follow the links.

        for i in 0..head.size {
            let next = pool.leaves[current as usize].next;

            assert!((next as usize) < leaf_limit);

            let current_prev = pool.leaves[current as usize].prev;

            if current_prev != prev {
                println!("verify_list:  {}:  current_prev {} != prev {} at {}",
                    name, current_prev, prev, i);

                return false;
            }

            prev    = current;
            current = next;

            assert!((current as usize) < leaf_limit);
            assert!((prev    as usize) < leaf_limit);
        }

        if current != first as Index {
            println!("verify_list:  The head doesn't match ({} vs {})",
                current, first);

            return false;
        }

        if pool.leaves[first].prev != prev {
            println!("verify_list:  The tail doesn't match ({} vs {})",
                pool.leaves[first].prev, prev);

            return false;
        }

        true
    }

    #[test]
    fn test_verify_list() {
        let mut pool  = get_simple_pool();
        let     index = get_non_empty_freelist(&pool).unwrap();
        let     list  = &pool.free_lists[index];
        let     first = list.first;
        let     tail  = pool.leaves[first].prev as usize;

        assert!(verify_pool(&pool));

        // The head of a zero length list should be an invalid id.
        // Set the size to zero and check the consistency.

        {
            let saved_size = list.size;
            pool.free_lists[index].size = 0;

            assert!(!verify_pool(&pool));

            pool.free_lists[index].size = saved_size;
            assert!(verify_pool(&pool));
        }

        // Now crunch a next pointer.

        {
            let saved_next = pool.leaves[first].next;
            pool.leaves[first].next = first as Index;

            assert!(!verify_pool(&pool));

            pool.leaves[first].next = saved_next;
            assert!(verify_pool(&pool));
        }

        // Now crunch a prev pointer.

        {
            let saved_prev = pool.leaves[first].prev;
            pool.leaves[first].prev = first as Index;

            assert!(!verify_pool(&pool));

            pool.leaves[first].prev = saved_prev;
            assert!(verify_pool(&pool));
        }

        // Crunch the tail's next.

        {
            pool.leaves[tail].next = tail as Index;

            assert!(!verify_pool(&pool));

            pool.leaves[tail].next = first as Index;
            assert!(verify_pool(&pool));
        }

        // Crunch the tail's prev.

        {
            let saved_prev = pool.leaves[tail as usize].prev;
            pool.leaves[tail].prev = tail as Index;

            assert!(!verify_pool(&pool));

            pool.leaves[tail].prev = saved_prev;
            assert!(verify_pool(&pool));
        }

        // Corrupt an empty list.

        {
            assert!(index > 0);

            let empty = &mut pool.free_lists[index - 1];
            assert!(empty.size == 0);

            let saved_first = empty.first;
            empty.first = first;

            assert!(!verify_pool(&pool));

            let empty = &mut pool.free_lists[index - 1];

            empty.first = saved_first;
            assert!(verify_pool(&pool));
        }

        // Try losing some memory...

        {
            let leaf = pool.dequeue_free(index).unwrap().unwrap();

            assert!(!verify_pool(&pool));

            pool.enqueue_free(index, leaf).unwrap();
            assert!(verify_pool(&pool));
        }
    }

    fn get_freelist_sizes(pool: &BuddyPool) -> Vec<usize> {
        let mut result = Vec::new();

        for list in &pool.free_lists {
            result.push(list.size);
        }

        result
    }

    #[test]
    fn test_try_merge_allocated() {
        let mut pool       = get_simple_pool();
        let     index      = get_non_empty_freelist(&pool).unwrap();
        let     min_size   = pool.min_size;
        let mut addresses  = Vec::new();
        let     factor     = 4;

        assert!(index > 1);
        assert!(factor <= pool.max_size / pool.min_size);

        // Allocate some small blobs.

        let big_address = pool.alloc(min_size * 4).unwrap();

        for _i in 0..8 {
            let address = pool.alloc(min_size).unwrap();

            addresses.push(address);
        }

        let area_offset = big_address - pool.base;
        let leaf_id     = area_offset / min_size;
        let leaf        = &pool.leaves[leaf_id];

        assert!(leaf.state == LeafState::Allocated);
        assert!(pool.is_allocated(leaf.state));

        let buddy_id = pool.buddy_id(leaf_id, BuddyAction::Merge).unwrap();
        assert!(buddy_id == leaf_id + factor);

        {
            let buddy = &pool.leaves[buddy_id];
            assert!(buddy.state == LeafState::Allocated);
        }

        let result = pool.try_merge_buddies(leaf_id);
        assert!(result.is_none());

        let buddy_address = pool.to_address(buddy_id).unwrap();
        pool.free(buddy_address).unwrap();

        {
            let buddy = &pool.leaves[buddy_id];
            assert!(buddy.state == LeafState::Free);
            assert!(pool.buddy_id(leaf_id, BuddyAction::Merge).unwrap() == buddy_id);
        }

        let result = pool.try_merge_buddies(leaf_id);
        assert!(result.is_none());
    }

    #[test]
    fn test_buddy_corners() {
        // Create a pool with space at the end that doesn't
        // fit into the maximum size freelist.

        let     base       = 0x10000000;
        let     min_alloc  = 0;
        let     min_buddy  = 1024;
        let     max_alloc  = 8192;
        let     size       = min_buddy * (64 + 2);
        let     config     = BuddyConfig { base, size, min_alloc, min_buddy, max_alloc };
        let mut pool       = BuddyPool::new(config).unwrap();
        let     sizes      = get_freelist_sizes(&pool);

        let     index      = get_non_empty_freelist(&pool).unwrap();
        let     leaf       = pool.dequeue_free(index).unwrap().unwrap();
        let     buddy      = pool.buddy_id(leaf, BuddyAction::Split).unwrap();

        let result = pool.buddy_id(leaf, BuddyAction::Merge);
        assert!(result.is_none());

        // Split leaf once.

        pool.split_buddy(leaf);

        let result = pool.buddy_id(leaf, BuddyAction::Split);
        assert!(result.is_none());

        // Merge them back together.

        let result = pool.try_merge_buddies(leaf).unwrap();
        assert!(result == leaf);
        assert!(sizes[index] == pool.free_lists[index].size + 1);

        // Split again.

        pool.split_buddy(leaf);
        pool.remove_free (index - 1, buddy).unwrap();
        pool.enqueue_free(index - 1, leaf ).unwrap();

        // Merge once again.

        let result = pool.try_merge_buddies(buddy).unwrap();
        assert!(result == leaf);
        assert!(pool.leaves[leaf].is_off_lists());

        let result = pool.buddy_id(leaf, BuddyAction::Merge);
        assert!(result.is_none());

        // Split again and mark the buddy as allocated.

        pool.split_buddy(leaf);
        pool.remove_free(index - 1, buddy).unwrap();
        pool.leaves[buddy].state = LeafState::Allocated;

        let result = pool.try_merge_buddies(leaf);
        assert!(result.is_none());

        pool.enqueue_free(index - 1, buddy).unwrap();

        let result = pool.try_merge_buddies(leaf).unwrap();
        assert!(result == leaf);
        assert!(pool.leaves[leaf].current_index == index as Index);

        pool.enqueue_free(index, leaf).unwrap();

        assert!(verify_pool(&pool));
        check_pool_sizes(&pool, &sizes);
    }

    #[test]
    fn basic_test() {
        let config =
            BuddyConfig {
                base:       0x1_0000_0000,      // the assumed base address of the memory region
                size:       1024 * 1024,        // the size of the memory region in bytes
                min_alloc:  1,                  // the minimum allocation size allowed
                max_alloc:  17 * 1024,          // the largest allocation that will be allowed
                min_buddy:  4096,               // the minimum size buddy block that will be created
            };

        let mut pool    = BuddyPool::new(config).unwrap();
        let     address = pool.alloc(config.max_alloc).unwrap();

        pool.free(address).unwrap();

        // Try a double free.

        let result = pool.free(address);

        assert!(result.is_err());
        assert!(result.unwrap_err().error_type() == BuddyErrorType::FreeingFreeMemory);

        // Try an undersized allocation.

        let result = pool.alloc(config.min_alloc - 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().error_type() == BuddyErrorType::UndersizeAlloc);

        // Try an oversized allocation.

        let result = pool.alloc(config.max_alloc + 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().error_type() == BuddyErrorType::OversizeAlloc);

        assert!(verify_pool(&pool));

        // Now allocate until the pool is empty.

        let mut allocs = 0;

        loop {
           let result = pool.alloc(config.min_alloc);

           if result.is_err() {
              let error = result.unwrap_err();

              assert!(error.error_type() == BuddyErrorType::OutOfMemory);
              break;
           }

          allocs += 1;
        }

        assert!(allocs == config.size / config.min_buddy);
    }

    fn test_state(pool: &BuddyPool, state: LeafState) {
        match state {
            LeafState::Free => {
                assert!( pool.is_mergeable(state));
                assert!(!pool.is_allocated(state));
            }

            LeafState::Allocated => {
                assert!(!pool.is_mergeable(state));
                assert!( pool.is_allocated(state));
            }

            LeafState::Merged => {
                assert!(!pool.is_mergeable(state));
                assert!(!pool.is_allocated(state));
            }

            LeafState::Wanted => {
                assert!(!pool.is_mergeable(state));
                assert!( pool.is_allocated(state));
            }

            LeafState::WantedHead => {
                assert!(!pool.is_mergeable(state));
                assert!( pool.is_allocated(state));
            }

            LeafState::Locked => {
                assert!( pool.is_mergeable(state));
                assert!(!pool.is_allocated(state));
            }

            LeafState::LockedHead => {
                assert!( pool.is_mergeable(state));
                assert!(!pool.is_allocated(state));
            }
        }
    }

    #[test]
    fn test_predicates() {
        let pool = get_simple_pool();

        test_state(&pool, LeafState::Free      );
        test_state(&pool, LeafState::Allocated );
        test_state(&pool, LeafState::Merged    );
        test_state(&pool, LeafState::Wanted    );
        test_state(&pool, LeafState::WantedHead);
        test_state(&pool, LeafState::Locked    );
        test_state(&pool, LeafState::LockedHead);
    }
}
