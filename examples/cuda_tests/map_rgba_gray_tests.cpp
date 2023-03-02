#include "tests_include.hpp"


bool map_rgba_test(img::Image const& src, img::View const& dst)
{
    printf("map_rgba_test\n");

    auto width = src.width / 2;
    auto height = src.height;    

    auto r_src = make_range(width, height);
    auto r_dst = make_range(src.width, src.height);
    r_dst.x_end = src.width;
    r_dst.x_begin = r_dst.x_end - width;

    auto src_v = img::sub_view(src, r_src);
    auto dst_v = img::sub_view(dst, r_dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 4);

    auto dv_1 = img::make_view(width, height, buffer32);
    auto dv_2 = img::make_view(width, height, buffer32);
    auto dst_d = img::make_view_4(width, height, buffer16);

    img::copy_to_device(src_v, dv_1);

    img::map_rgba(dv_1, dst_d);
    img::map_rgba(dst_d, dv_2);

    img::copy_to_host(dv_2, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_rgb_test(img::Image const& src, img::View const& dst)
{
    printf("map_rgb_test\n");

    auto width = src.width;
    auto height = src.height / 2;    

    auto r_src = make_range(width, height);
    auto r_dst = make_range(src.width, src.height);
    r_dst.y_end = src.height;
    r_dst.y_begin = r_dst.y_end - height;

    auto src_v = img::sub_view(src, r_src);
    auto dst_v = img::sub_view(dst, r_dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 3);

    auto dv_1 = img::make_view(width, height, buffer32);
    auto dv_2 = img::make_view(width, height, buffer32);
    auto dst_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, dv_1);

    img::map_rgb(dv_1, dst_d);
    img::map_rgb(dst_d, dv_2);

    img::copy_to_host(dv_2, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}