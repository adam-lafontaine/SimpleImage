#include "tests_include.hpp"


bool map_hsv_test(img::Image const& src, img::View const& dst)
{
    printf("map_hsv_test\n");
    fill_green(dst);

    auto width = src.width;
    auto height = src.height;

    auto src_v = img::make_view(src);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto hsv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb_hsv(src_dv, hsv_d);
    img::map_hsv_rgb(hsv_d, dst_dv);

    img::copy_to_host(dst_dv, dst);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}