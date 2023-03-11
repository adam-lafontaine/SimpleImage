#include "tests_include.hpp"

bool copy_platform_view_test()
{
    auto title = "copy_platform_view_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name)
    { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    u32 width = vette.width;
    u32 height = vette.height;

    GrayImage caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    GrayImage caddy;
    caddy.width = width;
    caddy.height = height;

    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    Image image;
    img::create_image(image, width, height);
    img::copy(img::make_view(vette), img::make_view(image));
    write_image(image, "vette_2.bmp");

    GrayImage gray;
    img::create_image(gray, width, height);
    img::copy(img::make_view(caddy), img::make_view(gray));
    write_image(gray, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    img::destroy_image(gray);
    img::destroy_image(image);

    return true;
}


bool copy_platform_sub_view_test()
{
    auto title = "copy_platform_sub_view_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name)
    { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    u32 width = vette.width;
    u32 height = vette.height;

    GrayImage caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    GrayImage caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    auto sub_w = width / 2;
    auto sub_h = height / 2;

    Range2Du32 r{};
    r.x_begin = width / 4;
    r.x_end = r.x_begin + sub_w;
    r.y_begin = height / 4;
    r.y_end = r.y_begin + sub_h;

    Image image;
    img::create_image(image, sub_w, sub_h);

    GrayImage gray;
    img::create_image(gray, sub_w, sub_h);

    img::copy(img::sub_view(vette, r), img::make_view(image));
    img::copy(img::sub_view(caddy, r), img::make_view(gray));

    write_image(image, "vette_2.bmp");
    write_image(gray, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    img::destroy_image(gray);
    img::destroy_image(image);

    return true;
}


bool copy_tests()
{
    printf("\n*** copy tests ***\n");

    auto result =
        copy_platform_view_test() &&
        copy_platform_sub_view_test();

    if (result)
    {
        printf("copy tests OK\n");
    }
    return result;
}