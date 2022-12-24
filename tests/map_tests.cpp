#include "tests_include.hpp"


static bool map_test()
{
    auto title = "map_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };
    
    GrayImage vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;
    
    GrayImage caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    GrayImage caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    //auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    auto buffer = img::create_buffer(width * height * 2);

    auto view_vette = img::make_view_1(width, height, buffer);
    auto view_caddy = img::make_view_1(width, height, buffer);

    img::map(vette, view_vette);
    img::map(caddy_v, view_caddy);

    img::map(view_caddy, vette);
    write_image(vette, "vette_2.bmp");

    img::map(view_vette, caddy_v);
    write_image(caddy, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}


bool map_tests()
{
    printf("\n*** map tests ***\n");

    auto result = map_test();

    if (result)
    {
        printf("map tests OK\n");
    }
    
    return result;
}