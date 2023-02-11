#pragma once

#include "../defines.hpp"

#include <functional>


// region of interest in an image
class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;  // one past last x
	u32 y_begin;
	u32 y_end;   // one past last y
};


inline Range2Du32 make_range(u32 width, u32 height)
{
	Range2Du32 r{};

	r.x_begin = 0;
	r.y_begin = 0;
	r.x_end = width;
	r.y_end = height;

	return r;
}


template <class T>
inline Range2Du32 make_range(T const& c)
{
	return make_range(c.width, c.height);
}


template <typename T>
class Matrix1D
{
public:
	T* data_ = nullptr;
	u32 length = 0;
};


template <typename T>
class Matrix2D
{
public:
	T* data_ = nullptr;
	u32 width = 0;
	u32 height = 0;

#ifndef NDEBUG

	//~Matrix2D() { assert(!(bool)data_); }

#endif // !NDEBUG
};


namespace simage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;


#if IS_LITTLE_ENDIAN

	class RGBAu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};

#else

	class RGBAu8
	{
	public:
		u8 alpha;
		u8 blue;
		u8 green;
		u8 red;
	};

#endif
    
}


/* platform image */

namespace simage
{
    typedef union pixel_t
	{
		u8 channels[4] = {};

		u32 value;

		RGBAu8 rgba;

	} Pixel;


	template <typename T>
    class MatrixView
	{
	public:

		T* matrix_data = 0;
		u32 matrix_width = 0;

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


	using Image = Matrix2D<Pixel>;
	using View = MatrixView<Pixel>;

	using ImageGray = Matrix2D<u8>;
	using ViewGray = MatrixView<u8>;

	using Mat2Dr32 = Matrix2D<r32>;
}


/* camera */

namespace simage
{
#if IS_LITTLE_ENDIAN

	class YUV422
	{
	public:
		u8 u;
		u8 y1;
		u8 v;
		u8 y2;
	};


	class YUV2
	{
	public:
		u8 uv;
		u8 y;
	};


	class BGR
	{
	public:
		u8 blue;
		u8 green;
		u8 red;
	};

#else

	class YUV422
	{
	public:
		u8 y2;
		u8 v;
		u8 y1;
		u8 u;
	};


	class YUV2
	{
	public:
		u8 y;
		u8 uv;
	};

	class BGR
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
	};

#endif


	using ImageYUV = Matrix2D<YUV2>;
	using ViewYUV = MatrixView<YUV2>;

	using ImageBGR = Matrix2D<BGR>;
	using ViewBGR = MatrixView<BGR>;
}


/* to_pixel */

namespace simage
{
	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		Pixel p{};
		p.rgba.red = r;
		p.rgba.green = g;
		p.rgba.blue = b;
		p.rgba.alpha = a;

		return p;
	}


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b)
	{
		return to_pixel(r, g, b, 255);
	}


	constexpr inline Pixel to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}
}


/* create destroy */

namespace simage
{
	bool create_image(Image& image, u32 width, u32 height);

	bool create_image(ImageGray& image, u32 width, u32 height);

	bool create_image(ImageYUV& image, u32 width, u32 height);

	void destroy_image(Image& image);

	void destroy_image(ImageGray& image);

	void destroy_image(ImageYUV& image);
}


/* make_view */

namespace simage
{
	View make_view(Image const& image);

	ViewGray make_view(ImageGray const& image);

	ViewYUV make_view(ImageYUV const& image);

	ViewBGR make_view(ImageBGR const& image);
}


/* sub_view */

namespace simage
{
	View sub_view(Image const& image, Range2Du32 const& range);

	ViewGray sub_view(ImageGray const& image, Range2Du32 const& range);

	View sub_view(View const& view, Range2Du32 const& range);

	ViewGray sub_view(ViewGray const& view, Range2Du32 const& range);


	ViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& range);

	ViewBGR sub_view(ImageBGR const& camera_src, Range2Du32 const& range);
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color);

	void fill(ViewGray const& view, u8 gray);
}


/* copy */

namespace simage
{
	void copy(View const& src, View const& dst);

	void copy(ViewGray const& src, ViewGray const& dst);
}


/* map */

namespace simage
{
	void map(ViewGray const& src, View const& dst);

	void map_yuv(ViewYUV const& src, View const& dst);

	void map(ViewYUV const& src, ViewGray const& dst);

	void map(ViewBGR const& src, View const& dst);
}


/* histogram */

namespace simage
{
	constexpr u32 MAX_HIST_BINS = 256;


	class HistRGBr32
	{
	public:
		r32 R[MAX_HIST_BINS];
		r32 G[MAX_HIST_BINS];
		r32 B[MAX_HIST_BINS];
	};


	class HistHSVr32
	{
	public:
		r32 H[MAX_HIST_BINS];
		r32 S[MAX_HIST_BINS];
		r32 V[MAX_HIST_BINS];
	};


	class HistLCHr32
	{
	public:
		r32 L[MAX_HIST_BINS];
		r32 C[MAX_HIST_BINS];
		r32 H[MAX_HIST_BINS];
	};


	class HistYUVr32
	{
	public:
		r32 Y[MAX_HIST_BINS];
		r32 U[MAX_HIST_BINS];
		r32 V[MAX_HIST_BINS];
	};


	class Histogram12r32
	{
	public:

		union
		{
			struct
			{
				HistRGBr32 rgb;
				HistHSVr32 hsv;
				HistLCHr32 lch;
				HistYUVr32 yuv;
			};

			r32 list[12][MAX_HIST_BINS] = { 0 };
		};

		u32 n_bins = MAX_HIST_BINS;
	};


	void make_histograms(View const& src, Histogram12r32& dst);

	void make_histograms(ViewYUV const& src, Histogram12r32& dst);

	void make_histograms(ViewBGR const& src, Histogram12r32& dst);


	void make_histograms(View const& src, HistRGBr32& dst, u32 n_bins);

	void make_histograms(View const& src, HistHSVr32& dst, u32 n_bins);

	void make_histograms(View const& src, HistLCHr32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistYUVr32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistRGBr32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistHSVr32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistLCHr32& dst, u32 n_bins);


}


/* row begin */

namespace simage
{
	template <typename T>
	inline T* row_begin(Matrix2D<T> const& image, u32 y)
	{
		return image.data_ + (u64)(y * image.width);
	}


	template <typename T>
	inline T* row_begin(MatrixView<T> const& view, u32 y)
	{
		return view.matrix_data + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
	}
}


/* stb_simage.cpp */

namespace simage
{
	bool read_image_from_file(const char* img_path_src, Image& image_dst);

	bool read_image_from_file(const char* file_path_src, ImageGray& image_dst);


#ifndef SIMAGE_NO_WRITE

	bool write_image(Image const& image_src, const char* file_path_dst);

	bool write_image(ImageGray const& image_src, const char* file_path_dst);

#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE

	bool resize_image(Image const& image_src, Image& image_dst);

	bool resize_image(ImageGray const& image_src, ImageGray& image_dst);

#endif // !SIMAGE_NO_RESIZE
}


/* read write */

#ifndef SIMAGE_NO_FILESYSTEM

#include <filesystem>


namespace simage
{
	using path_t = std::filesystem::path;


	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}


	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}


	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}

#endif // !SIMAGE_NO_WRITE

}

#else

#include <string>

namespace simage
{
	using path_t = std::string;

	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}


	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}


	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}

#endif // !SIMAGE_NO_WRITE

}

#endif // !SIMAGE_NO_FILESYSTEM



/* usb camera */

namespace simage
{
	class CameraUSB
	{
	public:
		int id = -1;
		u32 image_width = 0;
		u32 image_height = 0;
		u32 max_fps = 0;

		bool is_open;
	};


	using bgr_callback = std::function<void(ViewBGR const&)>;
	using view_callback = std::function<void(View const&)>;
	using bool_f = std::function<bool()>;


	bool open_camera(CameraUSB& camera);

	void close_camera(CameraUSB& camera);

	void close_all_cameras();

	bool grab_image(CameraUSB const& camera, View const& dst);

	bool grab_image(CameraUSB const& camera, view_callback const& callback);

	bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition);
}