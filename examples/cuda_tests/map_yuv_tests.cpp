#include "tests_include.hpp"


bool map_yuv_test(img::View const& src, img::View const& dst)
{
    printf("map_yuv_test\n");
    img::copy(src, dst);

    auto width = src.width;
    auto height = src.height/2;

    auto r = make_range(width, height);

    auto src_v = img::sub_view(src, r);
    auto dst_v = img::sub_view(dst, r);
    fill_green(dst_v);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto yuv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb_yuv(src_dv, yuv_d);
    img::map_yuv_rgb(yuv_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_yuv_channel_test(img::View const& src, img::View const& dst)
{
    printf("map_yuv_channel_test\n");
    img::copy(src, dst);
}