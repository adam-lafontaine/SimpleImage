#include "stb_include.hpp"
#include "../../simage.hpp"


#include <algorithm>
#include <cstring>


static bool has_extension(const char* filename, const char* ext)
{
	size_t file_length = std::strlen(filename);
	size_t ext_length = std::strlen(ext);

	return !std::strcmp(&filename[file_length - ext_length], ext);
}


static bool is_bmp(const char* filename)
{
	return has_extension(filename, ".bmp") || has_extension(filename, ".BMP");
}


static bool is_png(const char* filename)
{
	return has_extension(filename, ".png") || has_extension(filename, ".PNG");
}


namespace simage
{
    bool read_image_from_file(const char* img_path_src, Image& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 4;

		auto data = (Pixel*)stbi_load(img_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		if (!data)
		{
			return false;
		}

		image_dst.data_ = data;
		image_dst.width = width;
		image_dst.height = height;

		return true;
	}

#ifndef SIMAGE_NO_WRITE

	bool write_image(Image const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data_);

		int width = (int)(image_src.width);
		int height = (int)(image_src.height);
		int channels = (int)(RGBA_CHANNELS);
		auto const data = image_src.data_;

		int result = 0;

		if(is_bmp(file_path_dst))
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
			assert(result && " *** stbi_write_bmp() failed *** ");
		}
		else if(is_png(file_path_dst))
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
			assert(result && " *** stbi_write_png() failed *** ");
		}
		else
		{
			assert(false && " *** not a valid image format *** ");
		}

		return (bool)result;
	}

#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE

	bool resize_image(Image const& image_src, Image& image_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data_);
		assert(image_dst.width);
		assert(image_dst.height);

		int channels = (int)(RGBA_CHANNELS);

		auto layout = stbir_pixel_layout::STBIR_RGBA;

		int width_src = (int)(image_src.width);
		int height_src = (int)(image_src.height);
		int stride_bytes_src = width_src * channels;

		int width_dst = (int)(image_dst.width);
		int height_dst = (int)(image_dst.height);
		int stride_bytes_dst = width_dst * channels;

		auto data = stbir_resize_uint8_linear(
			(u8*)image_src.data_, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data_, width_dst, height_dst, stride_bytes_dst,
			layout);

		assert(data && " *** resize_image failed *** ");

		if (!image_dst.data_)
		{
			image_dst.data_ = (Pixel*)data;
		}

		return (bool)data;
	}

#endif // !SIMAGE_NO_RESIZE

	bool read_image_from_file(const char* file_path_src, ImageGray& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 1;

		auto data = (u8*)stbi_load(file_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		if (!data)
		{
			return false;
		}

		image_dst.data_ = data;
		image_dst.width = width;
		image_dst.height = height;

		return true;
	}

#ifndef SIMAGE_NO_WRITE

	bool write_image(ImageGray const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data_);

		int width = (int)(image_src.width);
		int height = (int)(image_src.height);
		int channels = 1;
		auto const data = image_src.data_;

		int result = 0;
		
		if(is_bmp(file_path_dst))
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
			assert(result && " *** stbi_write_bmp() failed *** ");
		}
		else if(is_png(file_path_dst))
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
			assert(result && " *** stbi_write_png() failed *** ");
		}
		else
		{
			assert(false && " *** not a valid image format *** ");
		}

		return (bool)result;
	}

#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE

	bool resize_image(ImageGray const& image_src, ImageGray& image_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data_);
		assert(image_dst.width);
		assert(image_dst.height);

		int channels = 1;

		auto layout = stbir_pixel_layout::STBIR_1CHANNEL;

		int width_src = (int)(image_src.width);
		int height_src = (int)(image_src.height);
		int stride_bytes_src = width_src * channels;

		int width_dst = (int)(image_dst.width);
		int height_dst = (int)(image_dst.height);
		int stride_bytes_dst = width_dst * channels;

		int result = 0;

		auto data = stbir_resize_uint8_linear(
			(u8*)image_src.data_, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data_, width_dst, height_dst, stride_bytes_dst,
			layout);

		assert(data && " *** resize_image gray failed *** ");

		if (!image_dst.data_)
		{
			image_dst.data_ = data;
		}

		return (bool)data;
	}

#endif // !SIMAGE_NO_RESIZE
}