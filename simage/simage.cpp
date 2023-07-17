#include "simage.hpp"
#include "src/util/execute.cpp"
#include "src/util/color_space.hpp"

#include <cmath>

namespace cs = color_space;


#include "src/impl/verify.cpp"
#include "src/impl/channel_pixels.cpp"
#include "src/impl/platform_image.cpp"
#include "src/impl/row_begin.cpp"
#include "src/impl/select_channel.cpp"
#include "src/impl/make_view.cpp"
#include "src/impl/sub_view.cpp"
#include "src/impl/fill.cpp"
#include "src/impl/convolve.cpp"
#include "src/impl/gradients.cpp"
#include "src/impl/blur.cpp"
#include "src/impl/rotate.cpp"
#include "src/impl/split_channels.cpp"
#include "src/impl/copy.cpp"
#include "src/impl/map_channels.cpp"
#include "src/impl/alpha_blend.cpp"
#include "src/impl/for_each_pixel.cpp"
#include "src/impl/transform.cpp"
#include "src/impl/centroid.cpp"
#include "src/impl/skeleton.cpp"
#include "src/impl/histogram.cpp"

#include "libs/stb/stb_simage.cpp"

#ifndef SIMAGE_NO_USB_CAMERA

#ifdef _WIN32

#include "libs/opencv/opencv_simage.cpp"

#else

#include "libs/uvc/uvc_simage.cpp"

#endif // _WIN32

#endif // !SIMAGE_NO_USB_CAMERA


#ifndef SIMAGE_NO_CUDA


/* reinterpret */

namespace simage
{
    template <typename T>
    static Matrix2D<T> reinterpret_matrix(DeviceMatrix2D<T> const& src)
    {
        Matrix2D<T> mat{};

        mat.data_ = src.data_;
        mat.width = src.width;
        mat.height = src.height;

        return mat;
    }
}

/* make_device_image */

namespace simage
{
	DeviceView make_device_view(u32 width, u32 height, DeviceBuffer32& buffer)
    {
        DeviceView image{};

        image.data_ = cuda::push_elements(buffer, width * height);
        image.width = width;
        image.height = height;

        return image;
    }


	DeviceViewGray make_device_view(u32 width, u32 height, DeviceBuffer8& buffer)
    {
        DeviceViewGray image{};

        image.data_ = cuda::push_elements(buffer, width * height);
        image.width = width;
        image.height = height;

        return image;
    }
}


/* copy device */

namespace simage
{
    template <typename T>
    static void do_copy_to_device(MatrixView<T> const& host_src, DeviceMatrix2D<T> const& device_dst)
    {
        auto const mat = reinterpret_matrix(device_dst);

        auto const bytes_per_row = sizeof(T) * host_src.width;

        for (u32 y = 0; y < host_src.height; ++y)
        {
            auto h = row_begin(host_src, y);
            auto d = row_begin(mat, y);
            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
        }
    }


    template <typename T>
    static void do_copy_to_host(DeviceMatrix2D<T> const& device_src, MatrixView<T> const& host_dst)
    {
        auto const mat = reinterpret_matrix(device_src);

        auto const bytes_per_row = sizeof(T) * device_src.width;

        for (u32 y = 0; y < host_dst.height; ++y)
        {
            auto d = row_begin(mat, y);
            auto h = row_begin(host_dst, y);
            if(!cuda::memcpy_to_host(d, h, bytes_per_row)) { assert(false); }
        }
    }


	void copy_to_device(View const& host_src, DeviceView const& device_dst)
    {
        assert(verify(host_src, device_dst));

        do_copy_to_device(host_src, device_dst);
    }


    void copy_to_device(ViewGray const& host_src, DeviceViewGray const& device_dst)
    {
        assert(verify(host_src, device_dst));

        do_copy_to_device(host_src, device_dst);
    }


    void copy_to_host(DeviceView const& device_src, View const& host_dst)
    {
        assert(verify(device_src, host_dst));

        do_copy_to_host(device_src, host_dst);
    }


    void copy_to_host(DeviceViewGray const& device_src, ViewGray const& host_dst)
    {
        assert(verify(device_src, host_dst));

        do_copy_to_host(device_src, host_dst);
    }
}

#endif // SIMAGE_NO_CUDA