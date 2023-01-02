#pragma once

#include "../defines.hpp"


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

	~Matrix2D() { assert(!(bool)data_); }

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

		T* image_data = 0;
		u32 image_width = 0;

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
}


/* camera */

namespace simage
{
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


	using ImageYUV = Matrix2D<YUV2>;
	using ViewYUV = MatrixView<YUV2>;
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
}


/* sub_view */

namespace simage
{
	View sub_view(Image const& image, Range2Du32 const& range);

	ViewGray sub_view(ImageGray const& image, Range2Du32 const& range);

	View sub_view(View const& view, Range2Du32 const& range);

	ViewGray sub_view(ViewGray const& view, Range2Du32 const& range);


	ViewYUV sub_view(ImageYUV const& camera_src, Range2Du32 const& image_range);
}


/* row begin */

namespace simage
{
	template <typename T>
	inline T* row_begin(Matrix2D<T> const& image, u32 y)
	{
		/*assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data_ + (u64)(offset);
		assert(ptr);

		return ptr;*/

		return image.data_ + (u64)(y * image.width);
	}


	template <typename T>
	inline T* row_begin(MatrixView<T> const& view, u32 y)
	{
		/*assert(verify(view));
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;*/

		return view.image_data + (u64)((view.y_begin + y) * view.image_width + view.x_begin);
	}
}