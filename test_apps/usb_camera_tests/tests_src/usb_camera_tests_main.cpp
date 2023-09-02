#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "USB Camera Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


bool init_camera_test_memory(u32 width, u32 height);
void destroy_camera_test_memory();


void grab_rgb_test(img::CameraUSB const& camera, img::View const& out);
void grab_gray_test(img::CameraUSB const& camera, img::View const& out);
void transform_test(img::CameraUSB const& camera, img::View const& out);
void threshold_min_max_test(img::CameraUSB const& camera, img::View const& out);
void binarize_test(img::CameraUSB const& camera, img::View const& out);
void alpha_blend_test(img::CameraUSB const& camera, img::View const& out);
void blur_rgb_test(img::CameraUSB const& camera, img::View const& out);
void gradients_tests(img::CameraUSB const& camera, img::View const& out);
void rotate_test(img::CameraUSB const& camera, img::View const& out);



static std::vector<std::function<void(img::CameraUSB const&, img::View const&)>> tests = 
{
    grab_rgb_test,
    grab_gray_test,
    transform_test,
    threshold_min_max_test,
    binarize_test,
    alpha_blend_test,
    blur_rgb_test,
    gradients_tests,
    rotate_test
};


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		true;
}


static void run_next_test(Input const& input, app::AppState& app_state, img::CameraUSB const& camera)
{
	app_state.check_for_stop();

    static int test_id = -1;
	static int frame_count = 0;

	frame_count++;   

	if (test_id < 0 && frame_count < FRAMES_PER_TEST)
	{
		return;
	}

    if (frame_count < FRAMES_PER_TEST)
    {
        tests[test_id](camera, app_state.screen_buffer[0]);
        return;
    }

	frame_count = 0;
	test_id++;	

	if (test_id >= tests.size())
	{
		app_state.stop();
	}	
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

    auto out_view = app_state.screen_buffer[0];

    if (!init_camera_test_memory(out_view.width, out_view.height))
    {
        img::close_camera(camera);
		return EXIT_FAILURE;
    }

	app_state.start();

    render_run(app_state, [&](auto const& input) { run_next_test(input, app_state, camera); });

    img::close_camera(camera);
    destroy_camera_test_memory();
    return EXIT_SUCCESS;
}