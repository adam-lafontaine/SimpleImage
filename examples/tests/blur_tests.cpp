#include "tests_include.hpp"


static bool blur_gray_test()
{
    auto title = "blur_gray_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name)
    { img::write_image(image, out_dir / name); };

    GrayImage chess;
    img::read_image_from_file(CHESS_PATH, chess);
    auto width = chess.width;
    auto height = chess.height;
    auto view = img::make_view(chess);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 2);

    auto src = img::make_view_1(width, height, buffer);
    auto dst = img::make_view_1(width, height, buffer);

    img::map_gray(view, src);

    write_image(chess, "chess.bmp");

    img::blur(src, dst);

    img::map_gray(dst, view);

    write_image(chess, "chess_blur.bmp");

    mb::reset_buffer(buffer);

    GrayImage caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);
    width = caddy.width;
    height = caddy.height;
    view = img::make_view(caddy);

    src = img::make_view_1(width, height, buffer);
    dst = img::make_view_1(width, height, buffer);

    img::map_gray(view, src);

    write_image(caddy, "caddy.bmp");

    img::blur(src, dst);

    img::map_gray(dst, view);

    write_image(caddy, "caddy_blur.bmp");

    img::destroy_image(chess);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}


static bool blur_rgb_test()
{
    auto title = "blur_rgb_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name)
    { img::write_image(image, out_dir / name); };

    Image chess;
    img::read_image_from_file(CHESS_PATH, chess);
    auto width = chess.width;
    auto height = chess.height;
    auto view = img::make_view(chess);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 6);

    auto src = img::make_view_3(width, height, buffer);
    auto dst = img::make_view_3(width, height, buffer);

    img::map_rgb(view, src);

    write_image(chess, "chess.bmp");

    img::blur(src, dst);

    img::map_rgb(dst, view);

    write_image(chess, "chess_blur.bmp");

    mb::reset_buffer(buffer);

    Image caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);
    width = caddy.width;
    height = caddy.height;
    view = img::make_view(caddy);

    src = img::make_view_3(width, height, buffer);
    dst = img::make_view_3(width, height, buffer);

    img::map_rgb(view, src);

    write_image(caddy, "caddy.bmp");

    img::blur(src, dst);

    img::map_rgb(dst, view);

    write_image(caddy, "caddy_blur.bmp");

    img::destroy_image(chess);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}



bool blur_tests()
{
    printf("\n*** blur tests ***\n");

    auto result =
        blur_gray_test() &&
        blur_rgb_test();

    if (result)
    {
        printf("blur tests OK\n");
    }

    return result;
}