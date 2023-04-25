licenses(["notice"])

cc_library(
    name = "acquire_retire",
    hdrs = ["acquire_retire.h"],
    deps = [
        "//parlay:parallel",
        "//parlay:portability",
        "//parlay:utilities",
    ],
)

cc_library(
    name = "hazptr_stack",
    hdrs = ["hazptr_stack.h"],
    deps = [
        ":acquire_retire",
    ],
)

package(
    default_visibility = ["//visibility:public"],
)
