#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"
#include "../util/execute.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <thread>


constexpr int N_CAMERAS = 1;
//constexpr u32 N_FRAMES = 2;


class CameraCV
{
public:
	cv::VideoCapture capture;
	cv::Mat frames[2];

	u32 frame_curr = 0;
	u32 frame_prev = 1;
};


static CameraCV g_cameras[N_CAMERAS];


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data;
	}


	static bool verify(CameraUSB const& camera)
	{
		return camera.image_width && camera.image_height && camera.max_fps && camera.id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.image_width == view.width &&
			camera.image_height == view.height;
	}
}

#endif



namespace simage
{
	static bool grab_current_frame(CameraCV& cam)
	{
		auto& cap = cam.capture;

		if (!cap.grab())
		{
			return false;
		}

		auto& frame = cam.frames[cam.frame_curr];

		if (!cap.retrieve(frame))
		{
			return false;
		}

		return true;
	}


	static void process_previous_frame(CameraCV& cam, std::function<void(ViewBGR const&)> const& grab_cb)
	{
		auto& frame = cam.frames[cam.frame_prev];

		ImageBGR image;
		image.width = (u32)frame.cols;
		image.height = (u32)frame.rows;
		image.data_ = (BGR*)frame.data;

		grab_cb(make_view(image));
	}


	bool open_camera(CameraUSB& camera)
	{
		auto& cap = g_cameras[0].capture;

		cap = cv::VideoCapture(1);
		if (!cap.isOpened())
		{
			cap = cv::VideoCapture(0);
			if (!cap.isOpened())
			{
				return false;
			}
		}

		camera.id = 0;
		camera.image_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.image_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);
		camera.is_open = true;

		assert(camera.image_width);
		assert(camera.image_height);
		assert(camera.max_fps);

		return true;
	}


	void close_camera(CameraUSB& camera)
	{
		camera.is_open = false;

		if (camera.id < 0 || camera.id >= N_CAMERAS)
		{
			return;
		}

		auto& camcv = g_cameras[camera.id];
		camcv.capture.release();
		camcv.frames[0].release();
		camcv.frames[1].release();

	}


	void close_all_cameras()
	{
		for (int i = 0; i < N_CAMERAS; ++i)
		{
			g_cameras[i].capture.release();

			g_cameras[i].frames[0].release();
			g_cameras[i].frames[1].release();
		}
	}


	bool grab_image(CameraUSB const& camera, View const& dst)
	{
		assert(verify(camera, dst));

		if (!camera.is_open || camera.id < 0 || camera.id >= N_CAMERAS)
		{
			return false;
		}

		auto& camcv = g_cameras[camera.id];

		if (!grab_current_frame(camcv))
		{
			return false;
		}

		auto& frame = camcv.frames[camcv.frame_curr];

		camcv.frame_curr = camcv.frame_curr == 0 ? 1 : 0;
		camcv.frame_prev = camcv.frame_curr == 0 ? 1 : 0;

		ImageBGR image;
		image.width = camera.image_width;
		image.height = camera.image_height;
		image.data_ = (BGR*)frame.data;

		map(make_view(image), dst);

		return true;
	}


	bool grab_image(CameraUSB const& camera, bgr_callback const& grab_cb)
	{
		if (!camera.is_open || camera.id < 0 || camera.id >= N_CAMERAS)
		{
			return false;
		}

		auto& camcv = g_cameras[camera.id];

		if (!grab_current_frame(camcv))
		{
			return false;
		}

		auto& frame = camcv.frames[camcv.frame_curr];

		camcv.frame_curr = camcv.frame_curr == 0 ? 1 : 0;
		camcv.frame_prev = camcv.frame_curr == 0 ? 1 : 0;

		ImageBGR image;
		image.width = camera.image_width;
		image.height = camera.image_height;
		image.data_ = (BGR*)frame.data;

		grab_cb(make_view(image));

		return true;
	}


	bool grab_continuous(CameraUSB const& camera, bgr_callback const& grab_cb, bool_f const& grab_condition)
	{
		if (!camera.is_open || camera.id < 0 || camera.id >= N_CAMERAS)
		{
			return false;
		}

		Stopwatch sw;
		auto target_ms_per_frame = (r64)camera.max_fps;
		auto frame_ms_elapsed = (r64)camera.max_fps;

		auto const wait_for_framerate = [&]()
		{
			frame_ms_elapsed = sw.get_time_milli();

			auto sleep_ms = (u32)(target_ms_per_frame - frame_ms_elapsed);
			if (frame_ms_elapsed < target_ms_per_frame && sleep_ms > 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
				while (frame_ms_elapsed < target_ms_per_frame)
				{
					frame_ms_elapsed = sw.get_time_milli();
				}
			}

			sw.start();
		};

		auto& camcv = g_cameras[camera.id];
		bool grab_ok[2] = { false, false };

		auto const grab_current = [&]() { grab_ok[camcv.frame_curr] = grab_current_frame(camcv); };

		auto const process_previous = [&]() 
		{ 
			if (grab_ok[camcv.frame_prev]) 
			{ 
				process_previous_frame(camcv, grab_cb);
			}			
		};

		std::array<std::function<void()>, 2> procs = 
		{
			grab_current, process_previous
		};

		sw.start();
		while (grab_condition())
		{
			execute(procs);

			camcv.frame_curr = camcv.frame_curr == 0 ? 1 : 0;
			camcv.frame_prev = camcv.frame_curr == 0 ? 1 : 0;

			wait_for_framerate();
		}

		return true;
	}

}