#include "tests_include.hpp"


bool map_rgba_test(img::View const& src, img::View const& dst)
{
    printf("map_rgba_test\n");

    img::copy(src, dst);

    auto width = src.width / 2;
    auto height = src.height;    

    auto r = make_range(width, height);

    auto src_v = img::sub_view(src, r);
    auto dst_v = img::sub_view(dst, r);
    fill_green(dst_v);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 4);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto dst_d = img::make_view_4(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgba(src_dv, dst_d);
    img::map_rgba(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_rgb_test(img::View const& src, img::View const& dst)
{
    printf("map_rgb_test\n");

    img::copy(src, dst);

    auto width = src.width;
    auto height = src.height / 2;    

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
    auto dst_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb(src_dv, dst_d);
    img::map_rgb(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}

