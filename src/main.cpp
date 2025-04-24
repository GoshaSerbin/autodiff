#include <exception>
#include <iostream>
#include <new>

#include "Logger.hpp"

auto main() -> int {
    try {
        LOG_INFO("hi, how are you?");
    } catch (const std::bad_alloc& e) {
        std::cerr << "Ошибка: нехватка памяти: " << e.what() << '\n';
    } catch (const std::ios_base::failure& e) {
        std::cerr << "Ошибка ввода/вывода: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Другая ошибка: " << e.what() << '\n';
    }
    return 0;
}
