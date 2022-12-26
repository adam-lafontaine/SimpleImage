#include "../src/util/memory_buffer.hpp"
#include "../src/defines.hpp"

#include <cstdio>

namespace mb = memory_buffer;


template <typename T>
static bool is_valid_ptr(T* ptr)
{
    return static_cast<bool>(ptr);
}


static bool create_destroy_test()
{
    printf("\ncreate_destroy_test\n");

    u32 n_elements = 100;
    bool result = false;

    MemoryBuffer<r32> buffer{};

    printf("create_buffer() - zero elements\n");
    result = !mb::create_buffer(buffer, 0);
    result &= !is_valid_ptr(buffer.data_);
    result &= (buffer.capacity_ == 0);
    result &= (buffer.size_ == 0);
    printf("data: %p\n", (void*)buffer.data_);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("create_buffer()\n");
    result = mb::create_buffer(buffer, n_elements);
    result &= is_valid_ptr(buffer.data_);
    result &= (buffer.capacity_ == n_elements);
    result &= (buffer.size_ == 0);
    printf("data: %p\n", (void*)buffer.data_);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("destroy_buffer()\n");
    mb::destroy_buffer(buffer);
    result = !is_valid_ptr(buffer.data_);
    result &= (buffer.capacity_ == 0);
    result &= (buffer.size_ == 0);
    printf("data: %p\n", (void*)buffer.data_);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);   
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    return true; 
}


static bool push_elements_test()
{
    printf("\npush_elements_test\n");

    u32 capacity = 100;
    size_t push = 25;
    bool result = false;

    MemoryBuffer<u32> buffer{};

    printf("create_buffer()\n");
    result = mb::create_buffer(buffer, capacity);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("push_elements() - zero elements\n");
    auto ptr = mb::push_elements(buffer, 0);
    result = !is_valid_ptr(ptr);
    printf("ptr: %p\n", (void*)ptr);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("push_elements()\n");
    auto chunk1 = mb::push_elements(buffer, push);
    result = is_valid_ptr(chunk1);
    result &= (buffer.size_ == push);
    printf("chunk1: %p\n", (void*)chunk1);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    auto chunk2 = mb::push_elements(buffer, push);
    auto ptr_diff = int(chunk2 - chunk1);
    result = is_valid_ptr(chunk2);
    result &= (buffer.size_ == 2 * push);
    result &= (ptr_diff >= push);
    printf("chunk2: %p\n", (void*)chunk2);
    printf("size: %zu\n", buffer.size_);
    printf("ptr_diff: %d\n", ptr_diff);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("push_elements() - too many elements\n");
    auto size = buffer.size_;
    auto chunk3 = mb::push_elements(buffer, buffer.capacity_ - buffer.size_ + 1);
    result = !is_valid_ptr(chunk3);
    result &= (buffer.size_ == size);
    result &= (buffer.capacity_ == capacity);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    mb::destroy_buffer(buffer);

    return true;
}


static bool reset_test()
{
    printf("\nreset_test\n");

    u32 n_elements = 100;
    size_t push = 75;
    bool result = false;

    MemoryBuffer<r32> buffer{};

    printf("create_buffer()\n");
    mb::create_buffer(buffer, n_elements);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);

    printf("push_elements()\n");
    auto chunk1 = mb::push_elements(buffer, push);
    printf("size: %zu\n", buffer.size_);

    printf("reset_buffer()\n");
    mb::reset_buffer(buffer);
    result = is_valid_ptr(buffer.data_);
    result &= (buffer.capacity_ == n_elements);
    result &= (buffer.size_ == 0);
    printf("data: %p\n", (void*)buffer.data_);
    printf("capacity: %zu\n", buffer.capacity_);
    printf("size: %zu\n", buffer.size_);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    mb::destroy_buffer(buffer);

    return true;
}


bool memory_buffer_tests()
{
    printf("\n*** memory_buffer tests ***\n");

    auto result = 
        create_destroy_test() &&
        push_elements_test() &&
        reset_test();

    if (result)
    {
        printf("memory_buffer tests OK\n");
    }
    return result;
}