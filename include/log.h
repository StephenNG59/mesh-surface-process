// log.h
#pragma once
#include <iostream>
#include <source_location>
#include <atomic>
#include <utility>

enum class LogLevel { error = 0, warning, info, debug, trace };

// 运行时保存当前级别：原子变量可保证多线程安全读取
inline std::atomic<LogLevel>& current_log_level() {
    static std::atomic<LogLevel> lvl{ LogLevel::info };
    return lvl;
}

// 把 "把若干参数依次 << 到 ostream" 封装成函数模板
template<class... Ts>
inline void log_write(std::ostream& os, Ts&&... xs) {
    (os << ... << std::forward<Ts>(xs));    // 真正的 fold-expression
}


#define LOG(LEVEL, ...)                                                          \
    do {                                                                         \
        if (LEVEL <= current_log_level().load(std::memory_order_relaxed)) {      \
            std::clog << '[' << #LEVEL << "] ";                                  \
            log_write(std::clog, ##__VA_ARGS__);                                 \
            std::clog << '\n';                                                   \
        }                                                                        \
    } while (0)

//<< std::source_location::current().file_name() << ':'      \
//<< std::source_location::current().line() << ' ';          \