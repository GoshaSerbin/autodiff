#include <chrono>
#include <sstream>
#include <string>

#include "Logger.hpp"

namespace ai {

    constexpr int MILLISECONDS_PER_SECOND = 1000;

    auto Logger::GetCurrentTime() -> std::string {
        auto now = std::chrono::system_clock::now();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % MILLISECONDS_PER_SECOND;
        const std::time_t time = std::chrono::system_clock::to_time_t(now);

        // The GetCurrentTime method is called only inside the Log method, which has mutex lock, therefore it is thread-safe
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        const std::tm tm = *std::localtime(&time);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << milliseconds.count();
        return oss.str();
    }

    auto Logger::GetInstance() noexcept -> Logger& {
        static Logger instance;
        return instance;
    }

    auto Logger::SetOutputStream(std::ostream* stream) noexcept -> void {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_outputStream = stream;
    }

    auto Logger::SetBuferization(bool buferization) noexcept -> void {
        m_buferization = buferization;
    }

    auto Logger::BuildPrefix(Level level, std::source_location location) -> std::string {
        std::ostringstream oss;
        oss << "[" << LevelToString(level) << "]\t" << "[" << GetCurrentTime() << "]\t" << "[" << location.function_name() << ", "
            << location.file_name() << ":" << location.line() << "]\t";
        return oss.str();
    }

    auto Logger::LevelToString(Level level) noexcept -> std::string {
        switch (level) {
            case Level::INFO:
                return "INFO";
            case Level::WARNING:
                return "WARNING";
            case Level::ERROR:
                return "ERROR";
            case Level::DEBUG:
                return "DEBUG";
            default:
                return "UNKNOWN";
        }
    }

}  // namespace ai
