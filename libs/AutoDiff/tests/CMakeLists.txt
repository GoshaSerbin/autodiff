add_executable(
    auto_diff_tests
    ScalarTest.cpp
    VectorTest.cpp
)

target_link_libraries(
    auto_diff_tests
    AutoDiff
    gtest
    gmock
    gcov
    GTest::gtest_main
)

add_test(NAME auto_diff_tests COMMAND auto_diff_tests)
