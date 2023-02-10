#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"
#include "../util/execute.hpp"

#include <array>
#include <thread>


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
    bool open_camera(CameraUSB& camera)
    {
        return false;
    }


    void close_all_cameras()
    {

    }


    bool grab_image(CameraUSB const& camera, View const& dst)
    {
        return false;
    }


    bool grab_image(CameraUSB const& camera, bgr_callback const& grab_cb)
    {
        return false;
    }


    bool grab_continuous(CameraUSB const& camera, bgr_callback const& grab_cb, bool_f const& grab_condition)
    {
        return false;
    }
}