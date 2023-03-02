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

		T* matrix_data_ = 0;
		u32 matrix_width = 0;

		u32 width = 0;
		u32 height = 0;

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

	class YUV422u8
	{
	public:
		u8 u;
		u8 y1;
		u8 v;
		u8 y2;
	};


	class YUV2u8
	{
	public:
		u8 uv;
		u8 y;
	};


	class BGRu8
	{
	public:
		u8 blue;
		u8 green;
		u8 red;
	};


	class RGBu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
	};


	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
	};


	enum class HSV : int
	{
		H = 0, S = 1, V = 2
	};


	enum class LCH : int
	{
		L = 0, C = 1, H = 2
	};


	enum class YUV : int
	{
		Y = 0, U = 1, V = 2
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};

#else

	class YUV422u8
	{
	public:
		u8 y2;
		u8 v;
		u8 y1;
		u8 u;
	};


	class YUV2u8
	{
	public:
		u8 y;
		u8 uv;
	};


	class BGRu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
	};


	class RGBu8
	{
	public:
		u8 blue;
		u8 green;
		u8 red;
	};


	enum class RGB : int
	{
		R = 2, G = 1, B = 0
	};


	enum class RGBA : int
	{
		R = 3, G = 2, B = 1, A = 0
	};


	enum class HSV : int
	{
		H = 2, S = 1, V = 0
	};


	enum class LCH : int
	{
		L = 2, C = 1, H = 0
	};


	enum class YUV : int
	{
		Y = 2, U = 1, V = 0
	};


	enum class GA : int
	{
		G = 1, A = 0
	};


	enum class XY : int
	{
		X = 1, Y = 0
	};

#endif	


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}


	using ImageYUV = Matrix2D<YUV2u8>;
	using ViewYUV = MatrixView<YUV2u8>;

	using ImageBGR = Matrix2D<BGRu8>;
	using ViewBGR = MatrixView<BGRu8>;

	using ImageRGB = Matrix2D<RGBu8>;
	using ViewRGB = MatrixView<RGBu8>;
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

	ViewRGB make_view(ImageRGB const& image);
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

	ViewBGR sub_view(ViewBGR const& camera_src, Range2Du32 const& range);

	ViewRGB sub_view(ImageRGB const& camera_src, Range2Du32 const& range);

	ViewRGB sub_view(ViewRGB const& camera_src, Range2Du32 const& range);
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
	void map_gray(View const& src, ViewGray const& dst);

	void map_gray(ViewGray const& src, View const& dst);

	void map_yuv(ViewYUV const& src, View const& dst);

	void map_gray(ViewYUV const& src, ViewGray const& dst);

	void map_rgb(ViewBGR const& src, View const& dst);

	void map_rgb(ViewRGB const& src, View const& dst);
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
		return view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
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
		int device_id = -1;
		u32 image_width = 0;
		u32 image_height = 0;
		u32 max_fps = 0;

		Image latest_frame;
		View frame_roi;

		bool is_open;
	};
	
	using view_callback = std::function<void(View const&)>;
	using bool_f = std::function<bool()>;


	bool open_camera(CameraUSB& camera);

	void close_camera(CameraUSB& camera);

	bool grab_image(CameraUSB const& camera);

	bool grab_image(CameraUSB const& camera, View const& dst);

	bool grab_image(CameraUSB const& camera, view_callback const& grab_cb);

	bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition);

	inline void set_roi(CameraUSB& camera, Range2Du32 const& roi) { camera.frame_roi = sub_view(camera.latest_frame, roi); }
}