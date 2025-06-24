# BuddyPool

a minimal buddy-system pool allocator

This library implements a simple buddy allocator.  The code does not assume
that the underlying memory (or other resource) is accessible directly by the
processor.  It thus can be used, for example to manage I/O memory that is
not mapped into the processor address space, or unmapped main memory.

As part of this approach, BuddyPool does not contain any unsafe code. It uses
Vec structures with integers for linking lists.
