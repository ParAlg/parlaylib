licenses(["notice"])

cc_library(
    name = "binary_search",
    hdrs = ["binary_search.h"],
)

cc_library(
    name = "block_allocator",
    hdrs = ["block_allocator.h"],
    deps = [
        ":memory_size",
        "//parlay:utilities",
        "//parlay:parallel",
        "//parlay/internal/concurrency:hazptr_stack",
    ],
)

cc_library(
    name = "block_delayed",
    hdrs = ["block_delayed.h"],
    deps = [
        ":get_time",
        ":sequence_ops",
        ":stream_delayed",
        "//parlay:delayed_sequence",
        "//parlay:monoid",
        "//parlay:parallel",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "bucket_sort",
    hdrs = ["bucket_sort.h"],
    deps = [
        ":merge_sort",
        ":quicksort",
        ":sequence_ops",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "collect_reduce",
    hdrs = ["collect_reduce.h"],
    deps = [
        ":counting_sort",
        ":get_time",
        ":integer_sort",
        ":sequence_ops",
        ":uninitialized_sequence",
        "//parlay:parallel",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "counting_sort",
    hdrs = ["counting_sort.h"],
    deps = [
        ":sequence_ops",
        ":uninitialized_sequence",
        "//parlay:monoid",
        "//parlay:parallel",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "debug_uninitialized",
    hdrs = ["debug_uninitialized.h"],
)

cc_library(
    name = "file_map",
    hdrs = ["file_map.h"],
    deps = [
        "//parlay/internal/posix:file_map_impl_posix",
        "//parlay/internal/windows:file_map_impl_windows",
    ],
)

cc_library(
    name = "get_time",
    hdrs = ["get_time.h"],
    deps = [],
)

cc_library(
    name = "group_by",
    hdrs = ["group_by.h"],
    deps = [
        ":block_delayed",
        ":collect_reduce",
        ":counting_sort",
        ":integer_sort",
        ":sample_sort",
        ":sequence_ops",
        "//parlay:monoid",
        "//parlay:primitives",
        "//parlay:range",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "heap_tree",
    hdrs = ["heap_tree.h"],
    deps = [
        "//parlay:sequence",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "integer_sort",
    hdrs = ["integer_sort.h"],
    deps = [
        ":counting_sort",
        ":get_time",
        ":quicksort",
        ":sequence_ops",
        ":uninitialized_sequence",
        "//parlay:delayed_sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "memory_size",
    hdrs = ["memory_size.h"],
)

cc_library(
    name = "merge",
    hdrs = ["merge.h"],
    deps = [
        ":binary_search",
        "//parlay:sequence",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "merge_sort",
    hdrs = ["merge_sort.h"],
    deps = [
        ":merge",
        ":quicksort",
        ":uninitialized_sequence",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "pool_allocator",
    hdrs = ["pool_allocator.h"],
    deps = [
        ":block_allocator",
        "//parlay/internal/concurrency:hazptr_stack",
    ],
)

cc_library(
    name = "quicksort",
    hdrs = ["quicksort.h"],
    deps = [
        ":counting_sort",
        ":sequence_ops",
        ":uninitialized_sequence",
        ":uninitialized_storage",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "sample_sort",
    hdrs = ["sample_sort.h"],
    deps = [
        ":bucket_sort",
        ":quicksort",
        ":sequence_ops",
        ":transpose",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "sequence_base",
    hdrs = ["sequence_base.h"],
    deps = [
        "//parlay:parallel",
        "//parlay:portability",
        "//parlay:type_traits",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "sequence_impl",
    hdrs = ["sequence_impl.h"],
    deps = [
        ":sequence_base",
        ":debug_uninitialized",
        "//parlay:alloc",
        "//parlay:parallel",
        "//parlay:range",
        "//parlay:slice",
        "//parlay:type_traits",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "sequence_ops",
    hdrs = ["sequence_ops.h"],
    deps = [
        "//parlay:delayed_sequence",
        "//parlay:monoid",
        "//parlay:parallel",
        "//parlay:range",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "stream_delayed",
    hdrs = ["stream_delayed.h"],
    deps = [
        ":uninitialized_sequence",
        "//parlay:sequence",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "transpose",
    hdrs = ["transpose.h"],
    deps = [
        ":sequence_ops",
        "//parlay:monoid",
        "//parlay:parallel",
        "//parlay:sequence",
        "//parlay:slice",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "uninitialized_sequence",
    hdrs = ["uninitialized_sequence.h"],
    deps = [
        ":debug_uninitialized",
        "//parlay:alloc",
    ],
)

cc_library(
    name = "uninitialized_storage",
    hdrs = ["uninitialized_storage.h"],
    deps = [
        ":debug_uninitialized",
    ],
)

cc_library(
    name = "work_stealing_job",
    hdrs = ["work_stealing_job.h"],
)

package(
    default_visibility = ["//visibility:public"],
)
