#include "../src/util/execute.hpp"
#include "../src/app/app.hpp"
#include "../render_usb/proc_def.hpp"

//#define LEAK_CHECK
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
#include "../src/util/win32_leak_check.h"
#endif

constexpr auto APP_TITLE = "Render USB";
constexpr auto APP_VERSION = "1.0";

static std::vector<std::function<void(img::View const&, img::View const&)>> proc_list =
{
	show_camera
};


static void run_selected_proc(Input const& input, img::CameraUSB const& camera, img::View const& dst)
{
	static int proc_id = -1;

	if (input.keyboard.space_key.pressed)
	{
		proc_id++;

		if (proc_id >= proc_list.size())
		{
			proc_id = 0;
		}
	}

	if (proc_id < 0 || proc_id >= proc_list.size())
	{
		return;
	}

	img::grab_image(camera, [&](img::View const& src) { proc_list[proc_id](src, dst); });
}


static img::View get_camera_view(img::CameraUSB const& camera, img::View const& app_screen)
{
	if (camera.image_width == app_screen.width && camera.image_height == app_screen.height)
	{
		return app_screen;
	}

	// assume camera is smaller than screen
	auto r = make_range(camera.image_width, camera.image_height);

	return img::sub_view(app_screen, r);
}


int main()
{
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
	int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
	dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
	dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
	_CrtSetDbgFlag(dbgFlags);
#endif

	app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = 1280;
	window_settings.screen_height = 720;

	app::AppState app_state;

	if (!render_init(window_settings, app_state))
	{
		return EXIT_FAILURE;
	}

	img::CameraUSB camera;
	if (!img::open_camera(camera))
	{
		return EXIT_FAILURE;
	}

	auto camera_out = get_camera_view(camera, app_state.screen_pixels);

	render_run(app_state, [&](auto const& input) { run_selected_proc(input, camera, camera_out); });

	img::close_camera(camera);

	return EXIT_SUCCESS;
}