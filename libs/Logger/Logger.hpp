#pragma once

#include <format>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <source_location>
#include <string>

namespace ai {

    /**
    @brief Static class for logging.

    Based on the singleton pattern, thread safety.
    The default stream is std::cout.
    It uses std::vformat for formating.
    For custom classes logging you need to implement specialization of std::formatter for your class.

    Setting output stream:
    \code
        auto file = std::make_unique<std::ofstream>("log.log");
        Logger::GetInstance().SetOutputStream(file.get());
    \endcode
    Custom class logging:
    \code
        class MyClass {
            public:
            int x = 42;
            int y = 42;
        };

        template <>
        struct std::formatter<MyClass> : std::formatter<std::string> {
            auto format(MyClass p, format_context& ctx) const { return formatter<string>::format(std::format("[{}, {}]", p.x, p.y),
    ctx); }
        };

        ...

        // logging
        LOG_INFO("Let me log pi={:10f}, str {}, e={:{}.{}f} and my class {} here!", pi, "hi", e, 3, 2, MyClass{});
        // [INFO]	[18:06:52.588]	[file.cpp, int main(), 23]	Let me log pi=  3.141593, str hi, e=2.72 and my class [42, 42] here!

        \endcode
    */
    class Logger {
     public:
        /// Log level.
        enum class Level {
            INFO,
            WARNING,
            ERROR,
            DEBUG
        };

        static auto GetInstance() noexcept -> Logger&;

        /// Sets the output stream for logger. E.g. file, std::cout, std::cerr.
        auto SetOutputStream(std::ostream* stream) noexcept -> void;

        /// Sets the buferization flag.
        auto SetBuferization(bool buferization) noexcept -> void;

        /// Logging with variadic arguments
        template <typename... Args>
        auto Log(Level level, std::source_location location, const std::string& message, Args&&... args) -> void {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (!m_outputStream) {
                return;
            }
            std::string prefix = BuildPrefix(level, location);
            std::string formattedMessage = std::vformat(prefix + message, std::make_format_args(args...));
            (*m_outputStream) << formattedMessage << "\n";
            if (m_buferization) {
                (*m_outputStream) << std::flush;
            }
        }

     private:
        std::ostream* m_outputStream = &std::cout;
        std::mutex m_mutex;
        bool m_buferization{false};

        Logger() = default;
        ~Logger() = default;

        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        static auto BuildPrefix(Level level, std::source_location location) -> std::string;

        static auto LevelToString(Level level) noexcept -> std::string;
    };

}  // namespace ai

#ifdef USE_ANDROID_LOGGING
#include <android/log.h>
#define LOG_TAG PROJECT_NAME
#define LOG_INFO(fmt, ...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "%s", std::format(fmt, __VA_ARGS__).c_str())
#define LOG_WARNING(fmt, ...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "%s", std::format(fmt, __VA_ARGS__).c_str())
#define LOG_DEBUG(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", std::format(fmt, __VA_ARGS__).c_str())
#define LOG_ERROR(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", std::format(fmt, __VA_ARGS__).c_str())
#else
/*!
 * @def LOG_INFO(fmt, ...)
 * @def LOG_WARNING(fmt, ...)
 * @def LOG_DEBUG(fmt, ...)
 * @def LOG_ERROR(fmt, ...)
 * @brief Макросы для логирования сообщений разных уровней.
 * @details
 * Эти макросы используются для логирования сообщений в зависимости от платформы:
 * - Если определен `USE_ANDROID_LOGGING`, используется `__android_log_print`.
 * - В противном случае используется `ai::Logger`.
 *
 * Уровни логирования:
 * - `LOG_INFO` — информационные сообщения.
 * - `LOG_WARNING` — предупреждающие сообщения.
 * - `LOG_DEBUG` — отладочные сообщения.
 * - `LOG_ERROR` — сообщения об ошибках.
 *
 * @param fmt Форматная строка (аналог `printf`).
 * @param ... Аргументы для подстановки в форматную строку.
 */
#define LOG_INFO(...) ai::Logger::GetInstance().Log(ai::Logger::Level::INFO, std::source_location::current(), __VA_ARGS__)
#define LOG_WARNING(...) ai::Logger::GetInstance().Log(ai::Logger::Level::WARNING, std::source_location::current(), __VA_ARGS__)
#define LOG_ERROR(...) ai::Logger::GetInstance().Log(ai::Logger::Level::ERROR, std::source_location::current(), __VA_ARGS__)
#define LOG_DEBUG(...) ai::Logger::GetInstance().Log(ai::Logger::Level::DEBUG, std::source_location::current(), __VA_ARGS__)
#endif
