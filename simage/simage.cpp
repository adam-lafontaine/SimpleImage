#include "src/simage_impl.cpp"
#include "libs/stb/stb_simage.cpp"
#include "src/util/execute.cpp"

#ifdef _WIN32

#include "libs/opencv/opencv_simage.cpp"

#else

#include "libs/uvc/uvc_simage.cpp"

#endif // _WIN32
