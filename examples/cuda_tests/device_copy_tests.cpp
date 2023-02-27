#include "tests_include.hpp"


using PixelBuffer = DeviceBuffer<img::Pixel>;


bool copy_image_test(img::Image const& src, img::View const& dst)
{
    printf("copy_image_test\n");

    auto width = src.width;
    auto height = src.height;

    PixelBuffer buffer;
    cuda::create_device_buffer(buffer, width * height);

    auto d_image = img::make_image(width, height, buffer);

    img::copy_to_device(src, d_image);
    img::copy_to_host(d_image, dst);

    cuda::destroy_buffer(buffer);

    return true;
}


bool copy_view_test(img::Image const& src, img::View const& dst)
{
    printf("copy_view_test\n");

    auto width = src.width;
    auto height = src.height;   

    auto src_r = make_range(width / 2, height / 2);
    
    auto src_v = img::sub_view(src, src_r);

    Range2Du32 dst_r{};
    dst_r.x_begin = width / 4;
    dst_r.x_end = dst_r.x_begin + width / 2;
    dst_r.y_begin = height / 4;
    dst_r.y_end = dst_r.y_begin + height / 2;

    auto dst_v = img::sub_view(dst, dst_r);

    PixelBuffer buffer;
    cuda::create_device_buffer(buffer, width * height / 4);

    auto d_image = img::make_image(width / 2, height / 2, buffer);

    img::copy_to_device(src_v, d_image);
    img::copy_to_host(d_image, dst_v);

    cuda::destroy_buffer(buffer);

    return true;
}