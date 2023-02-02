#include "tests_include.hpp"


template <typename T>
static bool is_valid_ptr(T* ptr)
{
    return static_cast<bool>(ptr);
}


template <class IMAGE>
static bool test_image_view(IMAGE const& image)
{
    bool result = false;

    auto view = img::make_view(image);
    printf("data: %p\n", (void*)view.matrix_data);
    printf("width: %u\n", view.width);
    printf("height: %u\n", view.height);

    result = is_valid_ptr(view.matrix_data);
    result &= (view.matrix_data == image.data_);
    result &= (view.width == image.width);
    result &= (view.height == image.height);
    result &= (view.x_begin == 0);
    result &= (view.y_begin == 0);
    result &= (view.x_end == image.width);
    result &= (view.y_end == image.height);

    return result;
}


static bool test_from_image()
{
    u32 width = 400;
    u32 height = 200;

    bool result = false;

    Image rgb;
    GrayImage gray;
    img::ImageYUV yuv;

    auto const cleanup = [&]()
    {
        img::destroy_image(rgb);
        img::destroy_image(gray);
        img::destroy_image(yuv);
    };

    printf("view rgb\n");    
    if (!img::create_image(rgb, width, height))
    {
        cleanup();
        return false;
    }
    result = test_image_view(rgb);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    printf("view gray\n");    
    if (!img::create_image(gray, width, height))
    {
        cleanup();
        return false;
    }
    result = test_image_view(gray);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    printf("view yuv\n");
    if (!img::create_image(yuv, width, height))
    {
        cleanup();
        return false;
    }
    result = test_image_view(yuv);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");    

    cleanup();

    return true;
}


static bool test_from_buffer()
{
    u32 width = 400;
    u32 height = 200;    

    auto const verify_view = [&](auto const& view)
    {
        bool result = false;
        printf("data: %p\n", (void*)view.image_channel_data[0]);
        printf("width: %u\n", view.width);
        printf("height: %u\n", view.height);
        result = is_valid_ptr(view.image_channel_data[0]);
        result &= (view.width == width);
        result &= (view.height == height);

        return result;
    };

    auto const verify_view_1 = [&](auto const& view)
    {
        bool result = false;
        printf("data: %p\n", (void*)view.matrix_data);
        printf("width: %u\n", view.width);
        printf("height: %u\n", view.height);
        result = is_valid_ptr(view.matrix_data);
        result &= (view.width == width);
        result &= (view.height == height);

        return result;
    };

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 10);

    auto const cleanup = [&](){ mb::destroy_buffer(buffer); };

    auto v1 = img::make_view_1(width, height, buffer);
    if (!verify_view_1(v1))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    auto v2 = img::make_view_2(width, height, buffer);
    if (!verify_view(v2))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    auto v3 = img::make_view_3(width, height, buffer);
    if (!verify_view(v3))
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

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


bool make_view_tests()
{
    printf("\n*** make_view tests ***\n");

    auto result = 
        test_from_image() &&
        test_from_buffer();

    if (result)
    {
        printf("make_view tests OK\n");
    }
    
    return result;
}