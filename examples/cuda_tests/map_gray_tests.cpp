#include "tests_include.hpp"


bool map_gray_test(img::View const& src, img::View const& dst)
{
    printf("map_gray_test\n");

    img::copy(src, dst);

    auto width = src.width / 2;
    auto height = src.height; 
    
    auto r = make_range(src.width, src.height);
    r.x_end = src.width;
    r.x_begin = r.x_end - width;

    auto src_v = img::sub_view(src, r);
    auto dst_v = img::sub_view(dst, r);
    fill_green(dst_v);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);    
    auto dst_d = img::make_view_1(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb_gray(src_dv, dst_d);
    img::map_gray_rgb(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);
    
    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}