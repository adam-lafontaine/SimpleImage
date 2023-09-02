#include "../../tests_include.hpp"


void gradients_tests(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);
    auto buffer32 = img::create_buffer32(width * height * 2);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer8);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(view, src);

    img::gradients(src, dst);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}


void gradients_xy_tests(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);
    auto buffer32 = img::create_buffer32(width * height * 3);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CHESS_PATH, image, width, height, buffer8);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_2(width, height, buffer32);

    auto grad_x = img::select_channel(dst, img::XY::X);
    auto grad_y = img::select_channel(dst, img::XY::Y);

    img::map_gray(view, src);

    img::gradients_xy(src, dst);

    auto to_abs = [](f32& p) { p = p < 0.0f ? p * -1.0f : p; };

    img::for_each_pixel(grad_x, to_abs);
    img::for_each_pixel(grad_y, to_abs);

    auto full = make_range(width, height);
    auto top = full;
    top.y_end = height / 2;
    auto bottom = full;
    bottom.y_begin = height / 2;

    img::map_rgba(img::sub_view(grad_x, top), img::sub_view(out, top));
    img::map_rgba(img::sub_view(grad_y, bottom), img::sub_view(out, bottom));

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}