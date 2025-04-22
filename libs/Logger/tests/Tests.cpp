#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "Logger.hpp"

using namespace ai;

class LoggerTests : public testing::Test {
 protected:
    void SetUp() override {
        output.clear();
        Logger::GetInstance().SetOutputStream(&output);
    }

    void TearDown() override {}

    std::ostringstream output;
};

TEST_F(LoggerTests, LogInfoMessage) {
    LOG_INFO("This is an info message");
    EXPECT_THAT(output.str(), ::testing::HasSubstr("This is an info message"));
    EXPECT_THAT(output.str(), ::testing::HasSubstr("INFO"));
}

TEST_F(LoggerTests, LogWarningMessage) {
    LOG_WARNING("This is a warning message");
    EXPECT_THAT(output.str(), ::testing::HasSubstr("This is a warning message"));
    EXPECT_THAT(output.str(), ::testing::HasSubstr("WARNING"));
}

TEST_F(LoggerTests, LogString) {
    LOG_INFO("Hello, {}!", "world");
    EXPECT_THAT(output.str(), ::testing::HasSubstr("Hello, world!"));
}

TEST_F(LoggerTests, LogNumbers) {
    LOG_INFO("Hello, {},{:5},{:*<5},{:*>5},{:*^6}", 42, 42, 42, 42, 42);
    EXPECT_THAT(output.str(), ::testing::HasSubstr("Hello, 42,   42,42***,***42,**42**"));
}

TEST_F(LoggerTests, LogFloatingPoints) {
    LOG_INFO("{:10.5f},{:.3f}", 3.14f, 3.14f);
    EXPECT_THAT(output.str(), ::testing::HasSubstr("   3.14000,3.140"));
}

TEST_F(LoggerTests, ToManyBracketsIsNotOK) {
    EXPECT_THROW(LOG_INFO("{} {}", "hello"), std::format_error);
}

TEST_F(LoggerTests, ToManyArgsIsOK) {
    LOG_INFO("{}", "hello", "world");
    EXPECT_THAT(output.str(), ::testing::HasSubstr("hello"));
}
