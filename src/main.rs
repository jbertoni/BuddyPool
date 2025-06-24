//
//  Copyright 2024 Jonathan L Bertoni
//
//  This code is available under the Berkeley 2-Clause, Berkeley 3-clause,
//  and MIT licenses.
//

use buddy_pool::BuddyPool;
use buddy_pool::BuddyConfig;
use buddy_pool::BuddyErrorType;
use std::alloc::alloc;
use std::alloc::Layout;

fn main() {
    // Allocate some memory for testing.

    let bytes: usize = 2 * 1024 * 1024;

    let malloc_base =
        unsafe {
            alloc(Layout::array::<u8>(bytes).unwrap())
        } as usize;

    let config =
        BuddyConfig {
            base:       malloc_base,        // the base address from libc::malloc
            size:       bytes,              // the size of the memory region in bytes
            min_alloc:  1,                  // the minimum allocation size allowed
            max_alloc:  17 * 1024,          // the largest allocation that will be allowed
            min_buddy:  8192,               // the minimum size buddy block that will be created
        };

    println!("Creating a pool at {:#x} containing {} bytes",
        config.base, config.size);

    let mut pool    = BuddyPool::new(config).unwrap();
    let     address = pool.alloc(config.max_alloc).unwrap();

    assert!(address >= config.base && address < config.base + config.size);

    pool.free(address).unwrap();

    println!("Allocated and freed at {:#x}", address);

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

    // Now allocate until the pool is empty.  Keep track
    // of the allocation count to see that it matches
    // the configuration.

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
