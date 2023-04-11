#pragma once

#include "src/util/memory_buffer.hpp"

#include <functional>

namespace mb = memory_buffer;


namespace simage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

	class RGBAu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};
    
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

	using Mat2Df32 = Matrix2D<f32>;
}


/* channel view */

namespace simage
{
	template <typename T, size_t N>
	class ChannelView2D
	{
	public:

		u32 channel_width_ = 0;

		T* channel_data_[N] = {};		

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
	

	template <typename T>
	using View4 = ChannelView2D<T, 4>;

	template <typename T>
	using View3 = ChannelView2D<T, 3>;

	template <typename T>
	using View2 = ChannelView2D<T, 2>;

	template <typename T>
	using View1 = MatrixView<T>;

	using View4u16 = View4<u16>;
	using View3u16 = View3<u16>;
	using View2u16 = View2<u16>;
	using View1u16 = View1<u16>;

	using View1u8 = ViewGray;

	using ViewRGBAu16 = View4u16;
	using ViewRGBu16 = View3u16;
	using ViewHSVu16 = View3u16;
	using ViewLCHu16 = View3u16;

    using Buffer32 = MemoryBuffer<Pixel>;
    using Buffer16 = MemoryBuffer<u16>;
    using Buffer8 = MemoryBuffer<u8>;
}


/* camera types */

namespace simage
{
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

#if IS_LITTLE_ENDIAN

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
	

	View make_view_rgba(u32 width, u32 height, Buffer32& buffer);

	ViewGray make_view_gray(u32 width, u32 height, Buffer8& buffer);
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


/* split channels */

namespace simage
{
	void split_rgb(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue);

	void split_rgba(View const& src, ViewGray const& red, ViewGray const& green, ViewGray const& blue, ViewGray const& alpha);

	void split_hsv(View const& src, ViewGray const& hue, ViewGray const& sat, ViewGray const& val);
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


/* alpha blend */

namespace simage
{
	void alpha_blend(View const& src, View const& cur, View const& dst);

	void alpha_blend(View const& src, View const& cur_dst);
}


/* for_each_pixel */

namespace simage
{
	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func);

	void for_each_pixel(ViewGray const& view, std::function<void(u8&)> const& func);
}


/* transform */

namespace simage
{
	using pixel_to_pixel_f = std::function<Pixel(Pixel)>;

	using u8_to_u8_f = std::function<u8(u8 p)>;

	using pixel_to_u8_f = std::function<u8(Pixel)>;

	using pixel_to_bool_f = std::function<bool(Pixel)>;

	using u8_to_bool_f = std::function<bool(u8)>;


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func);

	void transform(ViewGray const& src, ViewGray const& dst, u8_to_u8_f const& func);

	void transform(View const& src, ViewGray const& dst, pixel_to_u8_f const& func);


	void threshold(ViewGray const& src, ViewGray const& dst, u8 min);

	void threshold(ViewGray const& src, ViewGray const& dst, u8 min, u8 max);


	void binarize(View const& src, ViewGray const& dst, pixel_to_bool_f const& func);

	void binarize(ViewGray const& src, ViewGray const& dst, u8_to_bool_f const& func);

}


/* blur */

namespace simage
{
    void blur(View const& src, View const& dst);

    void blur(ViewGray const& src, ViewGray const& dst);
}


/* gradients */

namespace simage
{
    void gradients(ViewGray const& src, ViewGray const& dst);

    void gradients_xy(ViewGray const& src, ViewGray const& dst_x, ViewGray const& dst_y);    
}


/* rotate */

namespace simage
{
	void rotate(View const& src, View const& dst, Point2Du32 origin, f32 rad);

	void rotate(ViewGray const& src, ViewGray const& dst, Point2Du32 origin, f32 rad);
}


namespace simage
{
	Point2Du32 centroid(ViewGray const& src);

	Point2Du32 centroid(ViewGray const& src, u8_to_bool_f const& func);	


	void skeleton(ViewGray const& src_dst);
}


/* histogram */

namespace simage
{
namespace hist
{
	constexpr u32 MAX_HIST_BINS = 256;


	class HistRGBf32
	{
	public:
		union
		{
			struct
			{
				f32 R[MAX_HIST_BINS];
				f32 G[MAX_HIST_BINS];
				f32 B[MAX_HIST_BINS];
			};

			f32 channels[3][MAX_HIST_BINS];	
		};
	};


	class HistHSVf32
	{
	public:
		union
		{
			struct
			{
				f32 H[MAX_HIST_BINS];
				f32 S[MAX_HIST_BINS];
				f32 V[MAX_HIST_BINS];
			};

			f32 channels[3][MAX_HIST_BINS];	
		};
	};


	class HistLCHf32
	{
	public:
		union
		{
			struct
			{
				f32 L[MAX_HIST_BINS];
				f32 C[MAX_HIST_BINS];
				f32 H[MAX_HIST_BINS];
			};

			f32 channels[3][MAX_HIST_BINS];	
		};
	};


	class HistYUVf32
	{
	public:
		union
		{
			struct
			{
				f32 Y[MAX_HIST_BINS];
				f32 U[MAX_HIST_BINS];
				f32 V[MAX_HIST_BINS];
			};

			f32 channels[3][MAX_HIST_BINS];	
		};
	};


	class Histogram12f32
	{
	public:

		union
		{
			struct
			{
				HistRGBf32 rgb;
				HistHSVf32 hsv;
				HistLCHf32 lch;
				HistYUVf32 yuv;
			};

			f32 list[12][MAX_HIST_BINS] = { 0 };
		};

		u32 n_bins = MAX_HIST_BINS;
	};


	void make_histograms(View const& src, Histogram12f32& dst);

	void make_histograms(ViewYUV const& src, Histogram12f32& dst);

	void make_histograms(ViewBGR const& src, Histogram12f32& dst);


	void make_histograms(View const& src, HistRGBf32& dst, u32 n_bins);

	void make_histograms(View const& src, HistHSVf32& dst, u32 n_bins);

	void make_histograms(View const& src, HistLCHf32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistYUVf32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistRGBf32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistHSVf32& dst, u32 n_bins);

	void make_histograms(ViewYUV const& src, HistLCHf32& dst, u32 n_bins);

}
}


/* read write */
/* stb_simage.cpp */

#ifndef SIMAGE_NO_FILESYSTEM
#include <filesystem>
#else
#include <string>
#endif // !SIMAGE_NO_FILESYSTEM

namespace simage
{
	bool read_image_from_file(const char* img_path_src, Image& image_dst);

	bool read_image_from_file(const char* file_path_src, ImageGray& image_dst);

	bool write_image(Image const& image_src, const char* file_path_dst);

	bool write_image(ImageGray const& image_src, const char* file_path_dst);

	bool resize_image(Image const& image_src, Image& image_dst);

	bool resize_image(ImageGray const& image_src, ImageGray& image_dst);

#ifndef SIMAGE_NO_FILESYSTEM

    using path_t = std::filesystem::path;


	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}

	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}

#else

    using path_t = std::string;

	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}

	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}

#endif // !SIMAGE_NO_FILESYSTEM
}


/* usb camera */
/* uvc_simage.cpp or opencv_simage.cpp */

namespace simage
{
	class CameraUSB
	{
	public:
		int device_id = -1;
		u32 frame_width = 0;
		u32 frame_height = 0;
		u32 max_fps = 0;

		Image frame_image;

		Range2Du32 roi;

		bool is_open;
	};
	
	using rgb_callback = std::function<void(View const&)>;
	using gray_callback = std::function<void(ViewGray const&)>;
	using bool_f = std::function<bool()>;


	bool open_camera(CameraUSB& camera);

	void close_camera(CameraUSB& camera);

	bool grab_rgb(CameraUSB const& camera, View const& dst);

	bool grab_rgb(CameraUSB const& camera, rgb_callback const& grab_cb);

	bool grab_rgb_continuous(CameraUSB const& camera, rgb_callback const& grab_cb, bool_f const& grab_condition);
	
	bool grab_gray(CameraUSB const& camera, ViewGray const& dst);

	bool grab_gray(CameraUSB const& camera, gray_callback const& grab_cb);

	bool grab_gray_continuous(CameraUSB const& camera, gray_callback const& grab_cb, bool_f const& grab_condition);

	void set_roi(CameraUSB& camera, Range2Du32 roi);
}


/* make_view */

namespace simage
{
	View1u16 make_view_1(u32 width, u32 height, Buffer16& buffer);

	View2u16 make_view_2(u32 width, u32 height, Buffer16& buffer);

	View3u16 make_view_3(u32 width, u32 height, Buffer16& buffer);

	View4u16 make_view_4(u32 width, u32 height, Buffer16& buffer);
}


/* sub_view */

namespace simage
{
	View4u16 sub_view(View4u16 const& view, Range2Du32 const& range);

	View3u16 sub_view(View3u16 const& view, Range2Du32 const& range);

	View2u16 sub_view(View2u16 const& view, Range2Du32 const& range);

	View1u16 sub_view(View1u16 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1u16 select_channel(ViewRGBAu16 const& view, RGBA channel);

	View1u16 select_channel(ViewRGBu16 const& view, RGB channel);

	View1u16 select_channel(ViewHSVu16 const& view, HSV channel);

	View1u16 select_channel(View2u16 const& view, GA channel);

	View1u16 select_channel(View2u16 const& view, XY channel);


	ViewRGBu16 select_rgb(ViewRGBAu16 const& view);
}


/* map_gray */

namespace simage
{
	void map_gray(View1u8 const& src, View1u16 const& dst);

	void map_gray(View1u16 const& src, View1u8 const& dst);

	void map_gray(ViewYUV const& src, View1u16 const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAu16 const& dst);

	void map_rgba(ViewRGBAu16 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBu16 const& dst);

	void map_rgb(ViewRGBu16 const& src, View const& dst);

	void map_rgb(View1u16 const& src, View const& dst);


	inline void map_rgba(Image const& src, ViewRGBAu16 const& dst)
	{
		map_rgba(make_view(src), dst);
	}


	inline void map_rgb(Image const& src, ViewRGBu16 const& dst)
	{
		map_rgb(make_view(src), dst);
	}


	inline void map_rgb(ViewRGBu16 const& src, Image const& dst)
	{
		map_rgb(src, make_view(dst));
	}
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVu16 const& dst);

	void map_hsv_rgb(ViewHSVu16 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBu16 const& src, ViewHSVu16 const& dst);	

	void map_hsv_rgb(ViewHSVu16 const& src, ViewRGBu16 const& dst);
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHu16 const& dst);

	void map_lch_rgb(ViewLCHu16 const& src, View const& dst);

	void map_rgb_lch(ViewRGBu16 const& src, ViewLCHu16 const& dst);

	void map_lch_rgb(ViewLCHu16 const& src, ViewRGBu16 const& dst);
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBu16 const& dst);	
}


/* map_bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBu16 const& dst);
}


/* fill */

namespace simage
{
	void fill(View4u16 const& view, Pixel color);

	void fill(View3u16 const& view, Pixel color);

	void fill(View1u16 const& view, u8 gray);
}


/* transform */

namespace simage
{
	void transform(View1u16 const& src, View1u16 const& dst, std::function<f32(f32)> const& func32);

	void transform(View2u16 const& src, View1u16 const& dst, std::function<f32(f32, f32)> const& func32);

	void transform(View3u16 const& src, View1u16 const& dst, std::function<f32(f32, f32, f32)> const& func32);

	
	inline void transform_gray(ViewRGBu16 const& src, View1u16 const& dst)
	{
		return transform(src, dst, [](f32 red, f32 green, f32 blue) { return 0.299f * red + 0.587f * green + 0.114f * blue; });
	}


	void threshold(View1u16 const& src, View1u16 const& dst, f32 min32);

	void threshold(View1u16 const& src, View1u16 const& dst, f32 min32, f32 max32);


	void binarize(View1u16 const& src, View1u16 const& dst, std::function<bool(f32)> func32);
}


/* alpha blend */

namespace simage
{
	void alpha_blend(ViewRGBAu16 const& src, ViewRGBu16 const& cur, ViewRGBu16 const& dst);
}


/* rotate */

namespace simage
{
	void rotate(View4u16 const& src, View4u16 const& dst, Point2Du32 origin, f32 rad);

	void rotate(View3u16 const& src, View3u16 const& dst, Point2Du32 origin, f32 rad);

	void rotate(View2u16 const& src, View2u16 const& dst, Point2Du32 origin, f32 rad);

	void rotate(View1u16 const& src, View1u16 const& dst, Point2Du32 origin, f32 rad);
}


/* blur */

namespace simage
{
	void blur(View1u16 const& src, View1u16 const& dst);

	void blur(View3u16 const& src, View3u16 const& dst);
}


/* gradients */

namespace simage
{
    void gradients(View1u16 const& src, View1u16 const& dst);

	void gradients_xy(View1u16 const& src, View2u16 const& xy_dst);
}


/* shrink */
/*
namespace simage
{
	void shrink(View1u16 const& src, View1u16 const& dst);

	void shrink(View3u16 const& src, View3u16 const& dst);

	void shrink(View1u8 const& src, View1u16 const& dst);

	void shrink(View const& src, ViewRGBu16 const& dst);
}
*/


/* verify */

namespace simage
{
#ifndef NDEBUG
	template <typename T>
	static bool verify(Matrix2D<T> const& image)
	{
		return image.width && image.height && image.data_;
	}


	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data_;
	}	


	template <typename T>
	static bool verify(MemoryBuffer<T> const& buffer, u32 n_elements)
	{
		return n_elements && (buffer.capacity_ - buffer.size_) >= n_elements;
	}


	template <typename T, size_t N>
	static bool verify(ChannelView2D<T,N> const& view)
	{
		return view.channel_width_ && view.width && view.height && view.channel_data_[0];
	}


	template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}


	template <class IMG>
	static bool verify(IMG const& image, Range2Du32 const& range)
	{
		return
			verify(image) &&
			range.x_begin < range.x_end &&
			range.y_begin < range.y_end &&
			range.x_begin < image.width &&
			range.x_end <= image.width &&
			range.y_begin < image.height &&
			range.y_end <= image.height;
	}

#endif // !NDEBUG
}