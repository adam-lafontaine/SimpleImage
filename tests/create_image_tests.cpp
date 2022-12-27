#include "tests_include.hpp"

template <typename T>
static bool is_valid_ptr(T* ptr)
{
    return static_cast<bool>(ptr);
}


template <class IMAGE>
static bool create_test(IMAGE& image)
{
    bool result = false;
    u32 width = 400;
    u32 height = 200;
    
    result = img::create_image(image, width, height);
    result &= is_valid_ptr(image.data_);
    result &= (image.width == width);
    result &= (image.height == height);
    printf("data: %p\n", (void*)image.data_);
    printf("width: %u\n", image.width);
    printf("height: %u\n", image.height);
    
    return result;
}


template <class IMAGE>
static bool destroy_test(IMAGE& image)
{
    bool result = false;

    img::destroy_image(image);
    result = !is_valid_ptr(image.data_);
    result &= (image.width == 0);
    result &= (image.height == 0);
    printf("data: %p\n", (void*)image.data_);
    printf("width: %u\n", image.width);
    printf("height: %u\n", image.height);

    return result;
}


static bool create_destroy_test()
{
    bool result = false;

    printf("make rgb\n");
    Image rgb;
    result = create_test(rgb);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("destroy rgb\n");
    result = destroy_test(rgb);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("make gray\n");
    GrayImage gray;
    result = create_test(gray);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("destroy gray\n");
    result = destroy_test(gray);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("make yuv\n");
    img::ImageYUV yuv;
    result = create_test(yuv);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    printf("destroy yuv\n");
    result = destroy_test(yuv);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

    return true;
}


bool create_image_tests()
{
    printf("\n*** make_image tests ***\n");

    auto result = create_destroy_test();

    if (result)
    {
        printf("make_image tests OK\n");
    }
    
    return result;
}