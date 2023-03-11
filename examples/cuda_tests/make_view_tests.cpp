#include "tests_include.hpp"


template <typename T>
static bool is_valid_ptr(T* ptr)
{
    return static_cast<bool>(ptr);
}


bool make_view_tests()
{
    printf("make_vew_tests\n");

    u32 width = 400;
    u32 height = 200;

    auto const verify_view = [&](auto const& view)
    {
        bool result = false;
        printf("data: %p\n", (void*)view.channel_data_[0]);
        printf("width: %u\n", view.width);
        printf("height: %u\n", view.height);
        result = is_valid_ptr(view.channel_data_[0]);
        result &= (view.width == width);
        result &= (view.height == height);

        return result;
    };

    auto const verify_view_1 = [&](auto const& view)
    {
        bool result = false;
        printf("data: %p\n", (void*)view.matrix_data_);
        printf("width: %u\n", view.width);
        printf("height: %u\n", view.height);
        result = is_valid_ptr(view.matrix_data_);
        result &= (view.width == width);
        result &= (view.height == height);

        return result;
    };


    img::DeviceBuffer32 pixel_buffer;
    cuda::create_device_buffer(pixel_buffer, width * height);

    printf("make_view ");
    auto v = img::make_view(width, height, pixel_buffer);
    if (!verify_view_1(v))
    {
        printf("FAIL\n");
        cuda::destroy_buffer(pixel_buffer);
        return false;
    }

    cuda::destroy_buffer(pixel_buffer);

    img::DeviceBuffer8 gray_buffer;
    cuda::create_device_buffer(gray_buffer, width * height);

    printf("make_view gray ");
    auto vg = img::make_view(width, height, gray_buffer);
    if (!verify_view_1(vg))
    {
        printf("FAIL\n");
        cuda::destroy_buffer(gray_buffer);
        return false;
    }

    cuda::destroy_buffer(gray_buffer);


    img::DeviceBuffer16 buffer;
    cuda::create_device_buffer(buffer, width * height * 10);

    auto const cleanup = [&](){ cuda::destroy_buffer(buffer); };

    printf("make_view_1 ");
    auto v1 = img::make_view_1(width, height, buffer);
    if (!verify_view_1(v1))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    printf("make_view_2 ");
    auto v2 = img::make_view_2(width, height, buffer);
    if (!verify_view(v2))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    printf("make_view_3 ");
    auto v3 = img::make_view_3(width, height, buffer);
    if (!verify_view(v3))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    printf("make_view_4 ");
    auto v4 = img::make_view_4(width, height, buffer);
    if (!verify_view(v4))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    cleanup();
    return true;
}