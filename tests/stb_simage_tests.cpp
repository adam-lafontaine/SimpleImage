#include "tests_include.hpp"


static bool read_write_image_test()
{
	auto title = "read_write_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	bool result = false;


	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	printf("read rgb\n");
	result = img::read_image_from_file(CORVETTE_PATH, image);    
    printf("data: %p\n", (void*)image.data_);
	printf("width: %u\n", image.width);
    printf("height: %u\n", image.height);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

	printf("write rgb\n");
	result = img::write_image(image, out_dir / "corvette.bmp");
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

	GrayImage gray;
	printf("read gray\n");
	result = img::read_image_from_file(CADILLAC_PATH, gray);
	printf("data: %p\n", (void*)gray.data_);
	printf("width: %u\n", gray.width);
    printf("height: %u\n", gray.height);
    if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

	printf("write gray\n");
	result = img::write_image(gray, out_dir / "cadillac_gray.bmp");
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
    printf("OK\n");

	img::destroy_image(image);
	img::destroy_image(gray);

	return true;
}


static bool resize_test()
{
	auto title = "resize_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	bool result = false;

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	printf("stretch vertical rgb\n");
	Image vertical;	
	vertical.width = width / 2;
	vertical.height = height * 2;
	result = img::resize_image(image, vertical);
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
	printf("OK\n");
	write_image(vertical, "vertical.bmp");

	printf("stretch horizontal rgb\n");
	Image horizontal;
	horizontal.width = width * 2;
	horizontal.height = height / 2;
	result = img::resize_image(image, horizontal);
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
	printf("OK\n");

	write_image(horizontal, "horizontal.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	printf("stretch vertical gray\n");
	GrayImage vertical_gray;
	vertical_gray.width = width / 2;
	vertical_gray.height = height * 2;
	result = img::resize_image(gray, vertical_gray);
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
	printf("OK\n");

	write_image(vertical_gray, "vertical_gray.bmp");

	printf("stretch horizontal gray\n");
	GrayImage horizontal_gray;
	horizontal_gray.width = width * 2;
	horizontal_gray.height = height / 2;
	result = img::resize_image(gray, horizontal_gray);
	if (!result)
    {
        printf("FAIL\n");
        return false;
    }
	printf("OK\n");

	write_image(horizontal_gray, "horizontal_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(vertical);
	img::destroy_image(horizontal);
	img::destroy_image(gray);
	img::destroy_image(vertical_gray);
	img::destroy_image(horizontal_gray);

	return true;
}


bool stb_simage_tests()
{
	printf("\n***stb_simage tests ***\n");

	auto result = 
		read_write_image_test() &&
		resize_test();

	if (result)
	{
		printf("stb_simage tests OK\n");
	}
	
	return result;
}