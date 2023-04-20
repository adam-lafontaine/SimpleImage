#include "../../simage.hpp"
#include "../../src/util/execute.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <array>

namespace img = simage;


constexpr int N_CAMERAS = 1;


class DeviceCV
{
public:
	cv::VideoCapture capture;
	cv::Mat bgr_frame;
	img::ImageBGR bgr_image;

	img::ImageGray gray_image;
};


static DeviceCV g_devices[N_CAMERAS];


/* verify */

#ifndef NDEBUG

namespace simage
{
	static bool verify(CameraUSB const& camera)
	{
		return camera.frame_width && camera.frame_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.frame_width == view.width &&
			camera.frame_height == view.height;
	}
}

#endif


static void close_all_cameras()
{
	for (int i = 0; i < N_CAMERAS; ++i)
	{
		g_devices[i].capture.release();

		g_devices[i].bgr_frame.release();
	}
}


static bool grab_and_convert_frame_bgr(DeviceCV& device)
{
	auto& cap = device.capture;

	if (!cap.grab())
	{
		return false;
	}

	auto& frame = device.bgr_frame;

	if (!cap.retrieve(frame))
	{
		return false;
	}

	device.bgr_image.data_ = (img::BGRu8*)frame.data;

	return true;
}


static bool grab_and_convert_frame_gray(DeviceCV& device)
{
	if (!grab_and_convert_frame_bgr(device))
	{
		return false;
	}

	auto const to_gray = [&](u32 i) 
	{
		auto src = device.bgr_image.data_[i];
		auto& dst = device.gray_image.data_[i];
		auto gray = 0.299f * src.red + 0.587f * src.green + 0.114f * src.blue;
		dst = (u8)(gray + 0.5f);
	};

	auto n_pixels = device.bgr_image.width * device.bgr_image.height;

	process_range(0, n_pixels, to_gray);

	return true;
}



namespace simage
{
	static bool camera_is_initialized(CameraUSB const& camera)
	{
		return camera.is_open
			&& camera.device_id >= 0
			&& camera.device_id < N_CAMERAS;
	}


	bool open_camera(CameraUSB& camera)
	{
		auto& device = g_devices[0];
		auto& cap = device.capture;

		cap = cv::VideoCapture(1);
		if (!cap.isOpened())
		{
			cap = cv::VideoCapture(0);
			if (!cap.isOpened())
			{
				return false;
			}
		}

		camera.device_id = 0;
		camera.frame_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.frame_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);

		if (!create_image(camera.frame_image, camera.frame_width, camera.frame_height))
		{
			cap.release();
			return false;
		}

		if (!create_image(device.gray_image, camera.frame_width, camera.frame_height))
		{
			cap.release();
			return false;
		}

		ImageBGR bgr;
		device.bgr_image.width = camera.frame_width;
		device.bgr_image.height = camera.frame_height;
		device.bgr_image.data_ = (BGRu8*)(12345);

		auto roi = make_range(camera.frame_width, camera.frame_height);
		set_roi(camera, roi);

		camera.is_open = true;

		assert(camera.frame_width);
		assert(camera.frame_height);
		assert(camera.max_fps);

		return true;
	}


	void close_camera(CameraUSB& camera)
	{
		camera.is_open = false;

		if (camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return;
		}
		
		destroy_image(camera.frame_image);

		auto& device = g_devices[camera.device_id];
		destroy_image(device.gray_image);

		close_all_cameras();
	}


	bool grab_rgb(CameraUSB const& camera, View const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		auto device_view = sub_view(device.bgr_image, camera.roi);

		map_rgb(device_view, dst);

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, rgb_callback const& grab_cb)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		auto device_view = sub_view(device.bgr_image, camera.roi);
		auto camera_view = sub_view(camera.frame_image, camera.roi);

		map_rgb(device_view, camera_view);
		grab_cb(camera_view);

        return true;
	}


	bool grab_rgb_continuous(CameraUSB const& camera, rgb_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		auto device_view = sub_view(device.bgr_image, camera.roi);
		auto camera_view = sub_view(camera.frame_image, camera.roi);

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				device_view = sub_view(device.bgr_image, camera.roi);
				map_rgb(device_view, camera_view);
				grab_cb(camera_view);
			}
		}

		return false;
	}


	bool grab_gray(CameraUSB const& camera, ViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_gray(device))
		{
			return false;
		}

		auto device_view = sub_view(device.gray_image, camera.roi);

		copy(device_view, dst);

		return true;
	}


	bool grab_gray(CameraUSB const& camera, gray_callback const& grab_cb)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_gray(device))
		{
			return false;
		}

		auto device_view = sub_view(device.gray_image, camera.roi);

		grab_cb(device_view);

		return true;
	}


	bool grab_gray_continuous(CameraUSB const& camera, gray_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		auto device_view = sub_view(device.gray_image, camera.roi);

		while (grab_condition())
		{
			if (grab_and_convert_frame_gray(device))
			{
				grab_cb(device_view);
			}
		}

		return true;
	}


	void set_roi(CameraUSB& camera, Range2Du32 roi)
	{
		if (roi.x_end <= camera.frame_image.width &&
			roi.x_begin < roi.x_end &&
			roi.y_end <= camera.frame_height &&
			roi.y_begin < roi.y_end)
		{
			camera.roi = roi;
		}
	}
}