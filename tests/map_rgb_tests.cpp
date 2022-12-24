#include "tests_include.hpp"


static bool map_rgb_test()
{
    auto title = "map_rgb_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;

    Image caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    Image caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    auto buffer = img::create_buffer(width * height * 3 * 2);

    auto view_vette = img::make_view_3(width, height, buffer);
    auto view_caddy = img::make_view_3(width, height, buffer);

    img::map_rgb(vette_v, view_vette);
    img::map_rgb(caddy_v, view_caddy);

    img::map_rgb(view_caddy, vette_v);
    write_image(vette, "vette_2.bmp");

    img::map_rgb(view_vette, caddy_v);
    write_image(caddy, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}


static bool map_rgba_test()
{
    auto title = "map_rgba_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;

    Image caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    Image caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    auto buffer = img::create_buffer(width * height * 4 * 2);

    auto view_vette = img::make_view_4(width, height, buffer);
    auto view_caddy = img::make_view_4(width, height, buffer);

    img::map_rgb(vette_v, view_vette);
    img::map_rgb(caddy_v, view_caddy);

    img::map_rgb(view_caddy, vette_v);
    write_image(vette, "vette_2.bmp");

    img::map_rgb(view_vette, caddy_v);
    write_image(caddy, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}



bool map_rgb_tests()
{
    printf("\n*** map_rgb tests ***\n");

    auto result = 
        map_rgb_test() &&
        map_rgba_test();

    if (result)
    {
        printf("map_rgb tests OK\n");
    }
    
    return result;


}