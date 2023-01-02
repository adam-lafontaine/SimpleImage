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


#if IS_BIG_ENDIAN

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