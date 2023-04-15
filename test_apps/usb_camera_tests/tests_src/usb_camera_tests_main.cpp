#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "USB Camera Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.75 seconds
constexpr int FRAMES_PER_TEST = 45;


void grab_rgb_test(img::CameraUSB const& camera, img::View const& out);
void grab_gray_test(img::CameraUSB const& camera, img::View const& out);



static std::vector<std::function<void(img::CameraUSB const&, img::View const&)>> tests = 
{
    grab_rgb_test,
    grab_gray_test,
};


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		true;
}


static void run_next_test(Input const& input, app::AppState& app_state, img::CameraUSB const& camera)
{
    static int test_id = -1;
	static int frame_count = 0;

	frame_count++;   

	if (test_id < 0 && frame_count < FRAMES_PER_TEST)
	{
		return;
	}

    if (frame_count < FRAMES_PER_TEST)
    {
        tests[test_id](camera, app_state.screen_pixels);
        return;
    }

	frame_count = 0;
	test_id++;	

	if (test_id >= tests.size())
	{
		app_state.signal_stop = true;
		return;
	}	
}


static void adjust_screen_views(img::CameraUSB& camera, img::View& app_screen)
{
	if (camera.frame_width == app_screen.width && camera.frame_height == app_screen.height)
	{
		return;
	}	

	// change camera roi if it is larger than the screen
	auto roi_camera = make_range(camera.frame_width, camera.frame_height);

	if (camera.frame_width > app_screen.width)
	{
		roi_camera.x_begin = (camera.frame_width - app_screen.width) / 2;
		roi_camera.x_end = roi_camera.x_begin + app_screen.width;
	}

	if (camera.frame_height > app_screen.height)
	{
		roi_camera.y_begin = (camera.frame_height - app_screen.height) / 2;
		roi_camera.y_end = roi_camera.y_begin + app_screen.height;
	}

	img::set_roi(camera, roi_camera);

	// screen view that fits in camera roi
	u32 x_adj_screen = 0;
	u32 y_adj_screen = 0;
	
	if (camera.frame_width < app_screen.width)
	{
		x_adj_screen = (app_screen.width - camera.frame_width) / 2;
	}
	
	if (camera.frame_height < app_screen.height)
	{
		y_adj_screen = (app_screen.height - camera.frame_height) / 2;
	}
	
	auto roi_screen = make_range(camera.frame_width, camera.frame_height);

	roi_screen.x_begin += x_adj_screen;
	roi_screen.x_end += x_adj_screen;
	roi_screen.y_begin += y_adj_screen;
	roi_screen.y_end += y_adj_screen;	

	app_screen = img::sub_view(app_screen, roi_screen);
}


int main()
{
    if (!run_preliminary_tests())
	{
		return EXIT_FAILURE;
	}

    img::CameraUSB camera;
	if (!img::open_camera(camera))
	{
		return EXIT_FAILURE;
	}

	app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = camera.frame_width;
	window_settings.screen_height = camera.frame_height;

	app::AppState app_state;

    if (!render_init(window_settings, app_state))
	{
        img::close_camera(camera);
		return EXIT_FAILURE;
	}

    auto out_view = app_state.screen_pixels;
    adjust_screen_views(camera, out_view);

    render_run(app_state, [&](auto const& input) { run_next_test(input, app_state, camera); });

    img::close_camera(camera);
    return EXIT_SUCCESS;
}