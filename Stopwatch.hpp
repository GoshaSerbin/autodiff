#pragma once

#include <chrono>
#include <cmath>
#include "Logger.hpp"

using Clock = std::chrono::high_resolution_clock;

class Stopwatch {
 public:
    auto Reset() -> void {
        m_numOfMeasurements = 0;
        m_sumOfMeasurements = 0;
        m_sumOfSquaredMeasurements = 0;
    }
    auto Start() -> void { m_startTime = Clock::now(); }
    auto Stop() -> void {
        m_stopTime = Clock::now();
        int64_t time = GetMeasuredTimeInMicroseconds();
        m_numOfMeasurements += 1;
        m_sumOfMeasurements += time;
        m_sumOfSquaredMeasurements += time * time;
    }
    auto GetMeasuredTimeInMicroseconds() const -> int64_t {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_stopTime - m_startTime).count();
    }

    auto GetAVG() -> int64_t { return m_sumOfMeasurements / m_numOfMeasurements; }
    auto GetSTD() -> int64_t {
        return std::sqrt((m_numOfMeasurements * m_sumOfSquaredMeasurements - m_sumOfMeasurements * m_sumOfMeasurements) /
                         (m_numOfMeasurements * (m_numOfMeasurements - 1)));
    }

 private:
    std::chrono::time_point<Clock> m_startTime;
    std::chrono::time_point<Clock> m_stopTime;
    int64_t m_numOfMeasurements{};
    int64_t m_sumOfSquaredMeasurements{};
    int64_t m_sumOfMeasurements{};
};
