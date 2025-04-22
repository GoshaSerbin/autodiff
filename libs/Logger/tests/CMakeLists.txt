add_executable(
    logger_tests
    Tests.cpp
)

target_link_libraries(
    logger_tests
    Logger
    gtest
    gmock
    gcov
    GTest::gtest_main
)

add_test(NAME logger_tests COMMAND logger_tests)
