#pragma once

// Фолбэк-реализация макроса FRIEND_TEST для доступа к приватным полям из тестов GTest.
// Гарантирует успешную компиляцию, открывая тестовому классу доступ к скрытым буферам.

#ifndef FRIEND_TEST
#define FRIEND_TEST(test_case_name, test_name) \
    friend class test_case_name##_##test_name##_Test
#endif