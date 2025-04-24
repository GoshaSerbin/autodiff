#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "Logger.hpp"

using ai::Logger;

const float pi = 3.14f;

class LoggerTests : public testing::Test {
 protected:
    void SetUp() override { Logger::GetInstance().SetOutputStream(&m_output); }

    void TearDown() override { m_output.clear(); }

    auto GetOutput() -> std::string { return m_output.str(); }

 private:
    std::ostringstream m_output;
};

TEST_F(LoggerTests, LogInfoMessage) {
    LOG_INFO("This is an info message");
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("This is an info message"));
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("INFO"));
}

TEST_F(LoggerTests, LogWarningMessage) {
    LOG_WARNING("This is a warning message");
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("This is a warning message"));
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("WARNING"));
}

TEST_F(LoggerTests, LogString) {
    LOG_INFO("Hello, {}!", "world");
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("Hello, world!"));
}

TEST_F(LoggerTests, LogNumbers) {
    LOG_INFO("Hello, {},{:5},{:*<5},{:*>5},{:*^6}", 42, 42, 42, 42, 42);
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("Hello, 42,   42,42***,***42,**42**"));
}

TEST_F(LoggerTests, LogFloatingPoints) {
    LOG_INFO("{:10.5f},{:.3f}", pi, pi);
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("   3.14000,3.140"));
}

TEST_F(LoggerTests, ToManyBracketsIsNotOK) {
    EXPECT_THROW(LOG_INFO("{} {}", "hello"), std::format_error);
}

TEST_F(LoggerTests, ToManyArgsIsOK) {
    LOG_INFO("{}", "hello", "world");
    EXPECT_THAT(GetOutput(), ::testing::HasSubstr("hello"));
}
