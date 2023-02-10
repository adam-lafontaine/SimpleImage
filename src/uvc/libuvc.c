#include "libuvc_internal.h"

#include <errno.h>
#include <jpeglib.h>
#include <setjmp.h>

#define _POSIX_C_SOURCE 199309L
#include <time.h>


#define STREAM_C
#ifdef STREAM_C

#ifdef _MSC_VER

#define DELTA_EPOCH_IN_MICROSECS  116444736000000000Ui64

// gettimeofday - get time of day for Windows;
// A gettimeofday implementation for Microsoft Windows;
// Public domain code, author "ponnada";
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag = 0;
    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);
        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;
        tmpres /= 10;
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }
    return 0;
}
#endif // _MSC_VER
uvc_frame_desc_t *uvc_find_frame_desc_stream(uvc_stream_handle_t *strmh,
    uint16_t format_id, uint16_t frame_id);
uvc_frame_desc_t *uvc_find_frame_desc(uvc_device_handle_t *devh,
    uint16_t format_id, uint16_t frame_id);
void *_uvc_user_caller(void *arg);
void _uvc_populate_frame(uvc_stream_handle_t *strmh);

static uvc_streaming_interface_t *_uvc_get_stream_if(uvc_device_handle_t *devh, int interface_idx);
static uvc_stream_handle_t *_uvc_get_stream_by_interface(uvc_device_handle_t *devh, int interface_idx);

struct format_table_entry {
  enum uvc_frame_format format;
  uint8_t abstract_fmt;
  uint8_t guid[16];
  int children_count;
  enum uvc_frame_format *children;
};

struct format_table_entry *_get_format_entry(enum uvc_frame_format format) {
  #define ABS_FMT(_fmt, _num, ...) \
    case _fmt: { \
    static enum uvc_frame_format _fmt##_children[] = __VA_ARGS__; \
    static struct format_table_entry _fmt##_entry = { \
      _fmt, 0, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, _num, _fmt##_children }; \
    return &_fmt##_entry; }

  #define FMT(_fmt, ...) \
    case _fmt: { \
    static struct format_table_entry _fmt##_entry = { \
      _fmt, 0, __VA_ARGS__, 0, NULL }; \
    return &_fmt##_entry; }

  switch(format) {
    /* Define new formats here */
    ABS_FMT(UVC_FRAME_FORMAT_ANY, 2,
      {UVC_FRAME_FORMAT_UNCOMPRESSED, UVC_FRAME_FORMAT_COMPRESSED})

    ABS_FMT(UVC_FRAME_FORMAT_UNCOMPRESSED, 8,
      {UVC_FRAME_FORMAT_YUYV, UVC_FRAME_FORMAT_UYVY, UVC_FRAME_FORMAT_GRAY8,
       UVC_FRAME_FORMAT_GRAY16, UVC_FRAME_FORMAT_NV12, UVC_FRAME_FORMAT_P010,
       UVC_FRAME_FORMAT_BGR, UVC_FRAME_FORMAT_RGB})
    FMT(UVC_FRAME_FORMAT_YUYV,
      {'Y',  'U',  'Y',  '2', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_UYVY,
      {'U',  'Y',  'V',  'Y', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_GRAY8,
      {'Y',  '8',  '0',  '0', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_GRAY16,
      {'Y',  '1',  '6',  ' ', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_NV12,
      {'N',  'V',  '1',  '2', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_P010,
      {'P',  '0',  '1',  '0', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_BGR,
      {0x7d, 0xeb, 0x36, 0xe4, 0x4f, 0x52, 0xce, 0x11, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70})
    FMT(UVC_FRAME_FORMAT_RGB,
        {0x7e, 0xeb, 0x36, 0xe4, 0x4f, 0x52, 0xce, 0x11, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70})
    FMT(UVC_FRAME_FORMAT_BY8,
      {'B',  'Y',  '8',  ' ', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_BA81,
      {'B',  'A',  '8',  '1', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_SGRBG8,
      {'G',  'R',  'B',  'G', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_SGBRG8,
      {'G',  'B',  'R',  'G', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_SRGGB8,
      {'R',  'G',  'G',  'B', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    FMT(UVC_FRAME_FORMAT_SBGGR8,
      {'B',  'G',  'G',  'R', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})
    ABS_FMT(UVC_FRAME_FORMAT_COMPRESSED, 2,
      {UVC_FRAME_FORMAT_MJPEG, UVC_FRAME_FORMAT_H264})
    FMT(UVC_FRAME_FORMAT_MJPEG,
      {'M',  'J',  'P',  'G'})
    FMT(UVC_FRAME_FORMAT_H264,
      {'H',  '2',  '6',  '4', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71})

    default:
      return NULL;
  }

  #undef ABS_FMT
  #undef FMT
}

static uint8_t _uvc_frame_format_matches_guid(enum uvc_frame_format fmt, uint8_t guid[16]) {
  struct format_table_entry *format;
  int child_idx;

  format = _get_format_entry(fmt);
  if (!format)
    return 0;

  if (!format->abstract_fmt && !memcmp(guid, format->guid, 16))
    return 1;

  for (child_idx = 0; child_idx < format->children_count; child_idx++) {
    if (_uvc_frame_format_matches_guid(format->children[child_idx], guid))
      return 1;
  }

  return 0;
}

static enum uvc_frame_format uvc_frame_format_for_guid(uint8_t guid[16]) {
  struct format_table_entry *format;
  enum uvc_frame_format fmt;

  for (fmt = 0; fmt < UVC_FRAME_FORMAT_COUNT; ++fmt) {
    format = _get_format_entry(fmt);
    if (!format || format->abstract_fmt)
      continue;
    if (!memcmp(format->guid, guid, 16))
      return format->format;
  }

  return UVC_FRAME_FORMAT_UNKNOWN;
}

/** @internal
 * Run a streaming control query
 * @param[in] devh UVC device
 * @param[in,out] ctrl Control block
 * @param[in] probe Whether this is a probe query or a commit query
 * @param[in] req Query type
 */
uvc_error_t uvc_query_stream_ctrl(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uint8_t probe,
    enum uvc_req_code req) {
  uint8_t buf[34];
  size_t len;
  uvc_error_t err;

  memset(buf, 0, sizeof(buf));

  if (devh->info->ctrl_if.bcdUVC >= 0x0110)
    len = 34;
  else
    len = 26;

  /* prepare for a SET transfer */
  if (req == UVC_SET_CUR) {
    SHORT_TO_SW(ctrl->bmHint, buf);
    buf[2] = ctrl->bFormatIndex;
    buf[3] = ctrl->bFrameIndex;
    INT_TO_DW(ctrl->dwFrameInterval, buf + 4);
    SHORT_TO_SW(ctrl->wKeyFrameRate, buf + 8);
    SHORT_TO_SW(ctrl->wPFrameRate, buf + 10);
    SHORT_TO_SW(ctrl->wCompQuality, buf + 12);
    SHORT_TO_SW(ctrl->wCompWindowSize, buf + 14);
    SHORT_TO_SW(ctrl->wDelay, buf + 16);
    INT_TO_DW(ctrl->dwMaxVideoFrameSize, buf + 18);
    INT_TO_DW(ctrl->dwMaxPayloadTransferSize, buf + 22);

    if (len == 34) {
      INT_TO_DW ( ctrl->dwClockFrequency, buf + 26 );
      buf[30] = ctrl->bmFramingInfo;
      buf[31] = ctrl->bPreferredVersion;
      buf[32] = ctrl->bMinVersion;
      buf[33] = ctrl->bMaxVersion;
      /** @todo support UVC 1.1 */
    }
  }

  /* do the transfer */
  err = libusb_control_transfer(
      devh->usb_devh,
      req == UVC_SET_CUR ? 0x21 : 0xA1,
      req,
      probe ? (UVC_VS_PROBE_CONTROL << 8) : (UVC_VS_COMMIT_CONTROL << 8),
      ctrl->bInterfaceNumber,
      buf, len, 0
  );

  if (err <= 0) {
    return err;
  }

  /* now decode following a GET transfer */
  if (req != UVC_SET_CUR) {
    ctrl->bmHint = SW_TO_SHORT(buf);
    ctrl->bFormatIndex = buf[2];
    ctrl->bFrameIndex = buf[3];
    ctrl->dwFrameInterval = DW_TO_INT(buf + 4);
    ctrl->wKeyFrameRate = SW_TO_SHORT(buf + 8);
    ctrl->wPFrameRate = SW_TO_SHORT(buf + 10);
    ctrl->wCompQuality = SW_TO_SHORT(buf + 12);
    ctrl->wCompWindowSize = SW_TO_SHORT(buf + 14);
    ctrl->wDelay = SW_TO_SHORT(buf + 16);
    ctrl->dwMaxVideoFrameSize = DW_TO_INT(buf + 18);
    ctrl->dwMaxPayloadTransferSize = DW_TO_INT(buf + 22);

    if (len == 34) {
      ctrl->dwClockFrequency = DW_TO_INT ( buf + 26 );
      ctrl->bmFramingInfo = buf[30];
      ctrl->bPreferredVersion = buf[31];
      ctrl->bMinVersion = buf[32];
      ctrl->bMaxVersion = buf[33];
      /** @todo support UVC 1.1 */
    }
    else
      ctrl->dwClockFrequency = devh->info->ctrl_if.dwClockFrequency;

    /* fix up block for cameras that fail to set dwMax* */
    if (ctrl->dwMaxVideoFrameSize == 0) {
      uvc_frame_desc_t *frame = uvc_find_frame_desc(devh, ctrl->bFormatIndex, ctrl->bFrameIndex);

      if (frame) {
        ctrl->dwMaxVideoFrameSize = frame->dwMaxVideoFrameBufferSize;
      }
    }
  }

  return UVC_SUCCESS;
}

/** @internal
 * Run a streaming control query
 * @param[in] devh UVC device
 * @param[in,out] ctrl Control block
 * @param[in] probe Whether this is a probe query or a commit query
 * @param[in] req Query type
 */
uvc_error_t uvc_query_still_ctrl(
  uvc_device_handle_t *devh,
  uvc_still_ctrl_t *still_ctrl,
  uint8_t probe,
  enum uvc_req_code req) {

  uint8_t buf[11];
  const size_t len = 11;
  uvc_error_t err;

  memset(buf, 0, sizeof(buf));

  if (req == UVC_SET_CUR) {
    /* prepare for a SET transfer */
    buf[0] = still_ctrl->bFormatIndex;
    buf[1] = still_ctrl->bFrameIndex;
    buf[2] = still_ctrl->bCompressionIndex;
    INT_TO_DW(still_ctrl->dwMaxVideoFrameSize, buf + 3);
    INT_TO_DW(still_ctrl->dwMaxPayloadTransferSize, buf + 7);
  }

  /* do the transfer */
  err = libusb_control_transfer(
      devh->usb_devh,
      req == UVC_SET_CUR ? 0x21 : 0xA1,
      req,
      probe ? (UVC_VS_STILL_PROBE_CONTROL << 8) : (UVC_VS_STILL_COMMIT_CONTROL << 8),
      still_ctrl->bInterfaceNumber,
      buf, len, 0
  );

  if (err <= 0) {
    return err;
  }

  /* now decode following a GET transfer */
  if (req != UVC_SET_CUR) {
    still_ctrl->bFormatIndex = buf[0];
    still_ctrl->bFrameIndex = buf[1];
    still_ctrl->bCompressionIndex = buf[2];
    still_ctrl->dwMaxVideoFrameSize = DW_TO_INT(buf + 3);
    still_ctrl->dwMaxPayloadTransferSize = DW_TO_INT(buf + 7);
  }

  return UVC_SUCCESS;
}

/** Initiate a method 2 (in stream) still capture
 * @ingroup streaming
 *
 * @param[in] devh Device handle
 * @param[in] still_ctrl Still capture control block
 */
uvc_error_t uvc_trigger_still(
    uvc_device_handle_t *devh,
    uvc_still_ctrl_t *still_ctrl) {
  uvc_stream_handle_t* stream;
  uvc_streaming_interface_t* stream_if;
  uint8_t buf;
  uvc_error_t err;

  /* Stream must be running for method 2 to work */
  stream = _uvc_get_stream_by_interface(devh, still_ctrl->bInterfaceNumber);
  if (!stream || !stream->running)
    return UVC_ERROR_NOT_SUPPORTED;

  /* Only method 2 is supported */
  stream_if = _uvc_get_stream_if(devh, still_ctrl->bInterfaceNumber);
  if(!stream_if || stream_if->bStillCaptureMethod != 2)
      return UVC_ERROR_NOT_SUPPORTED;

  /* prepare for a SET transfer */
  buf = 1;

  /* do the transfer */
  err = libusb_control_transfer(
      devh->usb_devh,
      0x21, //type set
      UVC_SET_CUR,
      (UVC_VS_STILL_IMAGE_TRIGGER_CONTROL << 8),
      still_ctrl->bInterfaceNumber,
      &buf, 1, 0);

  if (err <= 0) {
    return err;
  }

  return UVC_SUCCESS;
}

/** @brief Reconfigure stream with a new stream format.
 * @ingroup streaming
 *
 * This may be executed whether or not the stream is running.
 *
 * @param[in] strmh Stream handle
 * @param[in] ctrl Control block, processed using {uvc_probe_stream_ctrl} or
 *             {uvc_get_stream_ctrl_format_size}
 */
uvc_error_t uvc_stream_ctrl(uvc_stream_handle_t *strmh, uvc_stream_ctrl_t *ctrl) {
  uvc_error_t ret;

  if (strmh->stream_if->bInterfaceNumber != ctrl->bInterfaceNumber)
    return UVC_ERROR_INVALID_PARAM;

  /* @todo Allow the stream to be modified without restarting the stream */
  if (strmh->running)
    return UVC_ERROR_BUSY;

  ret = uvc_query_stream_ctrl(strmh->devh, ctrl, 0, UVC_SET_CUR);
  if (ret != UVC_SUCCESS)
    return ret;

  strmh->cur_ctrl = *ctrl;
  return UVC_SUCCESS;
}

/** @internal
 * @brief Find the descriptor for a specific frame configuration
 * @param stream_if Stream interface
 * @param format_id Index of format class descriptor
 * @param frame_id Index of frame descriptor
 */
static uvc_frame_desc_t *_uvc_find_frame_desc_stream_if(uvc_streaming_interface_t *stream_if,
    uint16_t format_id, uint16_t frame_id) {
 
  uvc_format_desc_t *format = NULL;
  uvc_frame_desc_t *frame = NULL;

  DL_FOREACH(stream_if->format_descs, format) {
    if (format->bFormatIndex == format_id) {
      DL_FOREACH(format->frame_descs, frame) {
        if (frame->bFrameIndex == frame_id)
          return frame;
      }
    }
  }

  return NULL;
}

uvc_frame_desc_t *uvc_find_frame_desc_stream(uvc_stream_handle_t *strmh,
    uint16_t format_id, uint16_t frame_id) {
  return _uvc_find_frame_desc_stream_if(strmh->stream_if, format_id, frame_id);
}

/** @internal
 * @brief Find the descriptor for a specific frame configuration
 * @param devh UVC device
 * @param format_id Index of format class descriptor
 * @param frame_id Index of frame descriptor
 */
uvc_frame_desc_t *uvc_find_frame_desc(uvc_device_handle_t *devh,
    uint16_t format_id, uint16_t frame_id) {
 
  uvc_streaming_interface_t *stream_if;
  uvc_frame_desc_t *frame;

  DL_FOREACH(devh->info->stream_ifs, stream_if) {
    frame = _uvc_find_frame_desc_stream_if(stream_if, format_id, frame_id);
    if (frame)
      return frame;
  }

  return NULL;
}

/** Get a negotiated streaming control block for some common parameters.
 * @ingroup streaming
 *
 * @param[in] devh Device handle
 * @param[in,out] ctrl Control block
 * @param[in] format_class Type of streaming format
 * @param[in] width Desired frame width
 * @param[in] height Desired frame height
 * @param[in] fps Frame rate, frames per second
 */
uvc_error_t uvc_get_stream_ctrl_format_size(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    enum uvc_frame_format cf,
    int width, int height,
    int fps) {
  uvc_streaming_interface_t *stream_if;

  /* find a matching frame descriptor and interval */
  DL_FOREACH(devh->info->stream_ifs, stream_if) {
    uvc_format_desc_t *format;

    DL_FOREACH(stream_if->format_descs, format) {
      uvc_frame_desc_t *frame;

      if (!_uvc_frame_format_matches_guid(cf, format->guidFormat))
        continue;

      DL_FOREACH(format->frame_descs, frame) {
        if (frame->wWidth != width || frame->wHeight != height)
          continue;

        uint32_t *interval;

        ctrl->bInterfaceNumber = stream_if->bInterfaceNumber;
        UVC_DEBUG("claiming streaming interface %d", stream_if->bInterfaceNumber );
        uvc_claim_if(devh, ctrl->bInterfaceNumber);
        /* get the max values */
        uvc_query_stream_ctrl( devh, ctrl, 1, UVC_GET_MAX);

        if (frame->intervals) {
          for (interval = frame->intervals; *interval; ++interval) {
            // allow a fps rate of zero to mean "accept first rate available"
            if (10000000 / *interval == (unsigned int) fps || fps == 0) {

              ctrl->bmHint = (1 << 0); /* don't negotiate interval */
              ctrl->bFormatIndex = format->bFormatIndex;
              ctrl->bFrameIndex = frame->bFrameIndex;
              ctrl->dwFrameInterval = *interval;

              goto found;
            }
          }
        } else {
          uint32_t interval_100ns = 10000000 / fps;
          uint32_t interval_offset = interval_100ns - frame->dwMinFrameInterval;

          if (interval_100ns >= frame->dwMinFrameInterval
              && interval_100ns <= frame->dwMaxFrameInterval
              && !(interval_offset
                   && (interval_offset % frame->dwFrameIntervalStep))) {

            ctrl->bmHint = (1 << 0);
            ctrl->bFormatIndex = format->bFormatIndex;
            ctrl->bFrameIndex = frame->bFrameIndex;
            ctrl->dwFrameInterval = interval_100ns;

            goto found;
          }
        }
      }
    }
  }

  return UVC_ERROR_INVALID_MODE;

found:
  return uvc_probe_stream_ctrl(devh, ctrl);
}

/** Get a negotiated still control block for some common parameters.
 * @ingroup streaming
 *
 * @param[in] devh Device handle
 * @param[in] ctrl Control block
 * @param[in, out] still_ctrl Still capture control block
 * @param[in] width Desired frame width
 * @param[in] height Desired frame height
 */
uvc_error_t uvc_get_still_ctrl_format_size(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uvc_still_ctrl_t *still_ctrl,
    int width, int height) {
  uvc_streaming_interface_t *stream_if;
  uvc_still_frame_desc_t *still;
  uvc_format_desc_t *format;
  uvc_still_frame_res_t *sizePattern;

  stream_if = _uvc_get_stream_if(devh, ctrl->bInterfaceNumber);

  /* Only method 2 is supported */
  if(!stream_if || stream_if->bStillCaptureMethod != 2)
    return UVC_ERROR_NOT_SUPPORTED;

  DL_FOREACH(stream_if->format_descs, format) {

    if (ctrl->bFormatIndex != format->bFormatIndex)
      continue;

    /* get the max values */
    uvc_query_still_ctrl(devh, still_ctrl, 1, UVC_GET_MAX);

    //look for still format
    DL_FOREACH(format->still_frame_desc, still) {
      DL_FOREACH(still->imageSizePatterns, sizePattern) {

        if (sizePattern->wWidth != width || sizePattern->wHeight != height)
          continue;

        still_ctrl->bInterfaceNumber = ctrl->bInterfaceNumber;
        still_ctrl->bFormatIndex = format->bFormatIndex;
        still_ctrl->bFrameIndex = sizePattern->bResolutionIndex;
        still_ctrl->bCompressionIndex = 0; //TODO support compression index
        goto found;
      }
    }
  }

  return UVC_ERROR_INVALID_MODE;

  found:
    return uvc_probe_still_ctrl(devh, still_ctrl);
}

static int _uvc_stream_params_negotiated(
  uvc_stream_ctrl_t *required,
  uvc_stream_ctrl_t *actual) {
    return required->bFormatIndex == actual->bFormatIndex &&
    required->bFrameIndex == actual->bFrameIndex &&
    required->dwMaxPayloadTransferSize == actual->dwMaxPayloadTransferSize;
}

/** @internal
 * Negotiate streaming parameters with the device
 *
 * @param[in] devh UVC device
 * @param[in,out] ctrl Control block
 */
uvc_error_t uvc_probe_stream_ctrl(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl) {
  uvc_stream_ctrl_t required_ctrl = *ctrl;

  uvc_query_stream_ctrl( devh, ctrl, 1, UVC_SET_CUR );
  uvc_query_stream_ctrl( devh, ctrl, 1, UVC_GET_CUR );

  if(!_uvc_stream_params_negotiated(&required_ctrl, ctrl)) {
    UVC_DEBUG("Unable to negotiate streaming format");
    return UVC_ERROR_INVALID_MODE;
  }

  return UVC_SUCCESS;
}

/** @internal
 * Negotiate still parameters with the device
 *
 * @param[in] devh UVC device
 * @param[in,out] still_ctrl Still capture control block
 */
uvc_error_t uvc_probe_still_ctrl(
    uvc_device_handle_t *devh,
    uvc_still_ctrl_t *still_ctrl) {

  int res = uvc_query_still_ctrl(
    devh, still_ctrl, 1, UVC_SET_CUR
  );

  if(res == UVC_SUCCESS) {
    res = uvc_query_still_ctrl(
      devh, still_ctrl, 1, UVC_GET_CUR
    );

    if(res == UVC_SUCCESS) {
      res = uvc_query_still_ctrl(
        devh, still_ctrl, 0, UVC_SET_CUR
      );
    }
  }

  return res;
}

/** @internal
 * @brief Swap the working buffer with the presented buffer and notify consumers
 */
void _uvc_swap_buffers(uvc_stream_handle_t *strmh) {
  uint8_t *tmp_buf;

  pthread_mutex_lock(&strmh->cb_mutex);

  (void)clock_gettime(CLOCK_MONOTONIC, &strmh->capture_time_finished);

  /* swap the buffers */
  tmp_buf = strmh->holdbuf;
  strmh->hold_bytes = strmh->got_bytes;
  strmh->holdbuf = strmh->outbuf;
  strmh->outbuf = tmp_buf;
  strmh->hold_last_scr = strmh->last_scr;
  strmh->hold_pts = strmh->pts;
  strmh->hold_seq = strmh->seq;
  
  /* swap metadata buffer */
  tmp_buf = strmh->meta_holdbuf;
  strmh->meta_holdbuf = strmh->meta_outbuf;
  strmh->meta_outbuf = tmp_buf;
  strmh->meta_hold_bytes = strmh->meta_got_bytes;

  pthread_cond_broadcast(&strmh->cb_cond);
  pthread_mutex_unlock(&strmh->cb_mutex);

  strmh->seq++;
  strmh->got_bytes = 0;
  strmh->meta_got_bytes = 0;
  strmh->last_scr = 0;
  strmh->pts = 0;
}

/** @internal
 * @brief Process a payload transfer
 * 
 * Processes stream, places frames into buffer, signals listeners
 * (such as user callback thread and any polling thread) on new frame
 *
 * @param payload Contents of the payload transfer, either a packet (isochronous) or a full
 * transfer (bulk mode)
 * @param payload_len Length of the payload transfer
 */
void _uvc_process_payload(uvc_stream_handle_t *strmh, uint8_t *payload, size_t payload_len) {
  size_t header_len;
  uint8_t header_info;
  size_t data_len;

  /* magic numbers for identifying header packets from some iSight cameras */
  static uint8_t isight_tag[] = {
    0x11, 0x22, 0x33, 0x44,
    0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xfa, 0xce
  };

  /* ignore empty payload transfers */
  if (payload_len == 0)
    return;

  /* Certain iSight cameras have strange behavior: They send header
   * information in a packet with no image data, and then the following
   * packets have only image data, with no more headers until the next frame.
   *
   * The iSight header: len(1), flags(1 or 2), 0x11223344(4),
   * 0xdeadbeefdeadface(8), ??(16)
   */

  if (strmh->devh->is_isight &&
      (payload_len < 14 || memcmp(isight_tag, payload + 2, sizeof(isight_tag))) &&
      (payload_len < 15 || memcmp(isight_tag, payload + 3, sizeof(isight_tag)))) {
    /* The payload transfer doesn't have any iSight magic, so it's all image data */
    header_len = 0;
    data_len = payload_len;
  } else {
    header_len = payload[0];

    if (header_len > payload_len) {
      UVC_DEBUG("bogus packet: actual_len=%zd, header_len=%zd\n", payload_len, header_len);
      return;
    }

    if (strmh->devh->is_isight)
      data_len = 0;
    else
      data_len = payload_len - header_len;
  }

  if (header_len < 2) {
    header_info = 0;
  } else {
    /** @todo we should be checking the end-of-header bit */
    size_t variable_offset = 2;

    header_info = payload[1];

    if (header_info & 0x40) {
      UVC_DEBUG("bad packet: error bit set");
      return;
    }

    if (strmh->fid != (header_info & 1) && strmh->got_bytes != 0) {
      /* The frame ID bit was flipped, but we have image data sitting
         around from prior transfers. This means the camera didn't send
         an EOF for the last transfer of the previous frame. */
      _uvc_swap_buffers(strmh);
    }

    strmh->fid = header_info & 1;

    if (header_info & (1 << 2)) {
      strmh->pts = DW_TO_INT(payload + variable_offset);
      variable_offset += 4;
    }

    if (header_info & (1 << 3)) {
      /** @todo read the SOF token counter */
      strmh->last_scr = DW_TO_INT(payload + variable_offset);
      variable_offset += 6;
    }

    if (header_len > variable_offset) {
        // Metadata is attached to header
        size_t meta_len = header_len - variable_offset;
        if (strmh->meta_got_bytes + meta_len > LIBUVC_XFER_META_BUF_SIZE)
          meta_len = LIBUVC_XFER_META_BUF_SIZE - strmh->meta_got_bytes; /* Avoid overflow. */
        memcpy(strmh->meta_outbuf + strmh->meta_got_bytes, payload + variable_offset, meta_len);
        strmh->meta_got_bytes += meta_len;
    }
  }

  if (data_len > 0) {
    if (strmh->got_bytes + data_len > strmh->cur_ctrl.dwMaxVideoFrameSize)
      data_len = strmh->cur_ctrl.dwMaxVideoFrameSize - strmh->got_bytes; /* Avoid overflow. */
    memcpy(strmh->outbuf + strmh->got_bytes, payload + header_len, data_len);
    strmh->got_bytes += data_len;
    if (header_info & (1 << 1) || strmh->got_bytes == strmh->cur_ctrl.dwMaxVideoFrameSize) {
      /* The EOF bit is set, so publish the complete frame */
      _uvc_swap_buffers(strmh);
    }
  }
}

/** @internal
 * @brief Stream transfer callback
 *
 * Processes stream, places frames into buffer, signals listeners
 * (such as user callback thread and any polling thread) on new frame
 *
 * @param transfer Active transfer
 */
void LIBUSB_CALL _uvc_stream_callback(struct libusb_transfer *transfer) {
  uvc_stream_handle_t *strmh = transfer->user_data;

  int resubmit = 1;

  switch (transfer->status) {
  case LIBUSB_TRANSFER_COMPLETED:
    if (transfer->num_iso_packets == 0) {
      /* This is a bulk mode transfer, so it just has one payload transfer */
      _uvc_process_payload(strmh, transfer->buffer, transfer->actual_length);
    } else {
      /* This is an isochronous mode transfer, so each packet has a payload transfer */
      int packet_id;

      for (packet_id = 0; packet_id < transfer->num_iso_packets; ++packet_id) {
        uint8_t *pktbuf;
        struct libusb_iso_packet_descriptor *pkt;

        pkt = transfer->iso_packet_desc + packet_id;

        if (pkt->status != 0) {
          UVC_DEBUG("bad packet (isochronous transfer); status: %d", pkt->status);
          continue;
        }

        pktbuf = libusb_get_iso_packet_buffer_simple(transfer, packet_id);

        _uvc_process_payload(strmh, pktbuf, pkt->actual_length);

      }
    }
    break;
  case LIBUSB_TRANSFER_CANCELLED: 
  case LIBUSB_TRANSFER_ERROR:
  case LIBUSB_TRANSFER_NO_DEVICE: {
    int i;
    UVC_DEBUG("not retrying transfer, status = %d", transfer->status);
    pthread_mutex_lock(&strmh->cb_mutex);

    /* Mark transfer as deleted. */
    for(i=0; i < LIBUVC_NUM_TRANSFER_BUFS; i++) {
      if(strmh->transfers[i] == transfer) {
        UVC_DEBUG("Freeing transfer %d (%p)", i, transfer);
        free(transfer->buffer);
        libusb_free_transfer(transfer);
        strmh->transfers[i] = NULL;
        break;
      }
    }
    if(i == LIBUVC_NUM_TRANSFER_BUFS ) {
      UVC_DEBUG("transfer %p not found; not freeing!", transfer);
    }

    resubmit = 0;

    pthread_cond_broadcast(&strmh->cb_cond);
    pthread_mutex_unlock(&strmh->cb_mutex);

    break;
  }
  case LIBUSB_TRANSFER_TIMED_OUT:
  case LIBUSB_TRANSFER_STALL:
  case LIBUSB_TRANSFER_OVERFLOW:
    UVC_DEBUG("retrying transfer, status = %d", transfer->status);
    break;
  }
  
  if ( resubmit ) {
    if ( strmh->running ) {
      int libusbret = (uvc_error_t)libusb_submit_transfer(transfer);
      if (libusbRet < 0)
      {
        int i;
        pthread_mutex_lock(&strmh->cb_mutex);

        /* Mark transfer as deleted. */
        for (i = 0; i < LIBUVC_NUM_TRANSFER_BUFS; i++) {
          if (strmh->transfers[i] == transfer) {
            UVC_DEBUG("Freeing failed transfer %d (%p)", i, transfer);
            free(transfer->buffer);
            libusb_free_transfer(transfer);
            strmh->transfers[i] = NULL;
            break;
          }
        }
        if (i == LIBUVC_NUM_TRANSFER_BUFS) {
          UVC_DEBUG("failed transfer %p not found; not freeing!", transfer);
        }

        pthread_cond_broadcast(&strmh->cb_cond);
        pthread_mutex_unlock(&strmh->cb_mutex);
      }
    } else {
      int i;
      pthread_mutex_lock(&strmh->cb_mutex);

      /* Mark transfer as deleted. */
      for(i=0; i < LIBUVC_NUM_TRANSFER_BUFS; i++) {
        if(strmh->transfers[i] == transfer) {
          UVC_DEBUG("Freeing orphan transfer %d (%p)", i, transfer);
          free(transfer->buffer);
          libusb_free_transfer(transfer);
          strmh->transfers[i] = NULL;
          break;
        }
      }
      if(i == LIBUVC_NUM_TRANSFER_BUFS ) {
        UVC_DEBUG("orphan transfer %p not found; not freeing!", transfer);
      }

      pthread_cond_broadcast(&strmh->cb_cond);
      pthread_mutex_unlock(&strmh->cb_mutex);
    }
  }
}

/** Begin streaming video from the camera into the callback function.
 * @ingroup streaming
 *
 * @param devh UVC device
 * @param ctrl Control block, processed using {uvc_probe_stream_ctrl} or
 *             {uvc_get_stream_ctrl_format_size}
 * @param cb   User callback function. See {uvc_frame_callback_t} for restrictions.
 * @param flags Stream setup flags, currently undefined. Set this to zero. The lower bit
 * is reserved for backward compatibility.
 */
uvc_error_t uvc_start_streaming(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uvc_frame_callback_t *cb,
    void *user_ptr,
    uint8_t flags
) {
  uvc_error_t ret;
  uvc_stream_handle_t *strmh;

  ret = uvc_stream_open_ctrl(devh, &strmh, ctrl);
  if (ret != UVC_SUCCESS)
    return ret;

  ret = uvc_stream_start(strmh, cb, user_ptr, flags);
  if (ret != UVC_SUCCESS) {
    uvc_stream_close(strmh);
    return ret;
  }

  return UVC_SUCCESS;
}

/** Begin streaming video from the camera into the callback function.
 * @ingroup streaming
 *
 * @deprecated The stream type (bulk vs. isochronous) will be determined by the
 * type of interface associated with the uvc_stream_ctrl_t parameter, regardless
 * of whether the caller requests isochronous streaming. Please switch to
 * uvc_start_streaming().
 *
 * @param devh UVC device
 * @param ctrl Control block, processed using {uvc_probe_stream_ctrl} or
 *             {uvc_get_stream_ctrl_format_size}
 * @param cb   User callback function. See {uvc_frame_callback_t} for restrictions.
 */
uvc_error_t uvc_start_iso_streaming(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uvc_frame_callback_t *cb,
    void *user_ptr
) {
  return uvc_start_streaming(devh, ctrl, cb, user_ptr, 0);
}

static uvc_stream_handle_t *_uvc_get_stream_by_interface(uvc_device_handle_t *devh, int interface_idx) {
  uvc_stream_handle_t *strmh;

  DL_FOREACH(devh->streams, strmh) {
    if (strmh->stream_if->bInterfaceNumber == interface_idx)
      return strmh;
  }

  return NULL;
}

static uvc_streaming_interface_t *_uvc_get_stream_if(uvc_device_handle_t *devh, int interface_idx) {
  uvc_streaming_interface_t *stream_if;

  DL_FOREACH(devh->info->stream_ifs, stream_if) {
    if (stream_if->bInterfaceNumber == interface_idx)
      return stream_if;
  }
  
  return NULL;
}

/** Open a new video stream.
 * @ingroup streaming
 *
 * @param devh UVC device
 * @param ctrl Control block, processed using {uvc_probe_stream_ctrl} or
 *             {uvc_get_stream_ctrl_format_size}
 */
uvc_error_t uvc_stream_open_ctrl(uvc_device_handle_t *devh, uvc_stream_handle_t **strmhp, uvc_stream_ctrl_t *ctrl) {
  /* Chosen frame and format descriptors */
  uvc_stream_handle_t *strmh = NULL;
  uvc_streaming_interface_t *stream_if;
  uvc_error_t ret;

  UVC_ENTER();

  if (_uvc_get_stream_by_interface(devh, ctrl->bInterfaceNumber) != NULL) {
    ret = UVC_ERROR_BUSY; /* Stream is already opened */
    goto fail;
  }

  stream_if = _uvc_get_stream_if(devh, ctrl->bInterfaceNumber);
  if (!stream_if) {
    ret = UVC_ERROR_INVALID_PARAM;
    goto fail;
  }

  strmh = calloc(1, sizeof(*strmh));
  if (!strmh) {
    ret = UVC_ERROR_NO_MEM;
    goto fail;
  }
  strmh->devh = devh;
  strmh->stream_if = stream_if;
  strmh->frame.library_owns_data = 1;

  ret = uvc_claim_if(strmh->devh, strmh->stream_if->bInterfaceNumber);
  if (ret != UVC_SUCCESS)
    goto fail;

  ret = uvc_stream_ctrl(strmh, ctrl);
  if (ret != UVC_SUCCESS)
    goto fail;

  // Set up the streaming status and data space
  strmh->running = 0;

  strmh->outbuf = malloc( ctrl->dwMaxVideoFrameSize );
  strmh->holdbuf = malloc( ctrl->dwMaxVideoFrameSize );

  strmh->meta_outbuf = malloc( LIBUVC_XFER_META_BUF_SIZE );
  strmh->meta_holdbuf = malloc( LIBUVC_XFER_META_BUF_SIZE );
   
  pthread_mutex_init(&strmh->cb_mutex, NULL);
  pthread_cond_init(&strmh->cb_cond, NULL);

  DL_APPEND(devh->streams, strmh);

  *strmhp = strmh;

  UVC_EXIT(0);
  return UVC_SUCCESS;

fail:
  if(strmh)
    free(strmh);
  UVC_EXIT(ret);
  return ret;
}

/** Begin streaming video from the stream into the callback function.
 * @ingroup streaming
 *
 * @param strmh UVC stream
 * @param cb   User callback function. See {uvc_frame_callback_t} for restrictions.
 * @param flags Stream setup flags, currently undefined. Set this to zero. The lower bit
 * is reserved for backward compatibility.
 */
uvc_error_t uvc_stream_start(
    uvc_stream_handle_t *strmh,
    uvc_frame_callback_t *cb,
    void *user_ptr,
    uint8_t flags
) {
  /* USB interface we'll be using */
  const struct libusb_interface *interface;
  int interface_id;
  char isochronous;
  uvc_frame_desc_t *frame_desc;
  uvc_format_desc_t *format_desc;
  uvc_stream_ctrl_t *ctrl;
  uvc_error_t ret;
  /* Total amount of data per transfer */
  size_t total_transfer_size = 0;
  struct libusb_transfer *transfer;
  int transfer_id;

  ctrl = &strmh->cur_ctrl;

  UVC_ENTER();

  if (strmh->running) {
    UVC_EXIT(UVC_ERROR_BUSY);
    return UVC_ERROR_BUSY;
  }

  strmh->running = 1;
  strmh->seq = 1;
  strmh->fid = 0;
  strmh->pts = 0;
  strmh->last_scr = 0;

  frame_desc = uvc_find_frame_desc_stream(strmh, ctrl->bFormatIndex, ctrl->bFrameIndex);
  if (!frame_desc) {
    ret = UVC_ERROR_INVALID_PARAM;
    goto fail;
  }
  format_desc = frame_desc->parent;

  strmh->frame_format = uvc_frame_format_for_guid(format_desc->guidFormat);
  if (strmh->frame_format == UVC_FRAME_FORMAT_UNKNOWN) {
    ret = UVC_ERROR_NOT_SUPPORTED;
    goto fail;
  }

  // Get the interface that provides the chosen format and frame configuration
  interface_id = strmh->stream_if->bInterfaceNumber;
  interface = &strmh->devh->info->config->interface[interface_id];

  /* A VS interface uses isochronous transfers iff it has multiple altsettings.
   * (UVC 1.5: 2.4.3. VideoStreaming Interface) */
  isochronous = interface->num_altsetting > 1;

  if (isochronous) {
    /* For isochronous streaming, we choose an appropriate altsetting for the endpoint
     * and set up several transfers */
    const struct libusb_interface_descriptor *altsetting = 0;
    const struct libusb_endpoint_descriptor *endpoint = 0;
    /* The greatest number of bytes that the device might provide, per packet, in this
     * configuration */
    size_t config_bytes_per_packet;
    /* Number of packets per transfer */
    size_t packets_per_transfer = 0;
    /* Size of packet transferable from the chosen endpoint */
    size_t endpoint_bytes_per_packet = 0;
    /* Index of the altsetting */
    int alt_idx, ep_idx;
    
    config_bytes_per_packet = strmh->cur_ctrl.dwMaxPayloadTransferSize;

    /* Go through the altsettings and find one whose packets are at least
     * as big as our format's maximum per-packet usage. Assume that the
     * packet sizes are increasing. */
    for (alt_idx = 0; alt_idx < interface->num_altsetting; alt_idx++) {
      altsetting = interface->altsetting + alt_idx;
      endpoint_bytes_per_packet = 0;

      /* Find the endpoint with the number specified in the VS header */
      for (ep_idx = 0; ep_idx < altsetting->bNumEndpoints; ep_idx++) {
        endpoint = altsetting->endpoint + ep_idx;

        struct libusb_ss_endpoint_companion_descriptor *ep_comp = 0;
        libusb_get_ss_endpoint_companion_descriptor(NULL, endpoint, &ep_comp);
        if (ep_comp)
        {
          endpoint_bytes_per_packet = ep_comp->wBytesPerInterval;
          libusb_free_ss_endpoint_companion_descriptor(ep_comp);
          break;
        }
        else
        {
          if (endpoint->bEndpointAddress == format_desc->parent->bEndpointAddress) {
              endpoint_bytes_per_packet = endpoint->wMaxPacketSize;
            // wMaxPacketSize: [unused:2 (multiplier-1):3 size:11]
            endpoint_bytes_per_packet = (endpoint_bytes_per_packet & 0x07ff) *
              (((endpoint_bytes_per_packet >> 11) & 3) + 1);
            break;
          }
        }
      }

      if (endpoint_bytes_per_packet >= config_bytes_per_packet) {
        /* Transfers will be at most one frame long: Divide the maximum frame size
         * by the size of the endpoint and round up */
        packets_per_transfer = (ctrl->dwMaxVideoFrameSize +
                                endpoint_bytes_per_packet - 1) / endpoint_bytes_per_packet;

        /* But keep a reasonable limit: Otherwise we start dropping data */
        if (packets_per_transfer > 32)
          packets_per_transfer = 32;
        
        total_transfer_size = packets_per_transfer * endpoint_bytes_per_packet;
        break;
      }
    }

    /* If we searched through all the altsettings and found nothing usable */
    if (alt_idx == interface->num_altsetting) {
      ret = UVC_ERROR_INVALID_MODE;
      goto fail;
    }

    /* Select the altsetting */
    ret = (uvc_error_t)libusb_set_interface_alt_setting(strmh->devh->usb_devh,
                                           altsetting->bInterfaceNumber,
                                           altsetting->bAlternateSetting);
    if (ret != UVC_SUCCESS) {
      UVC_DEBUG("libusb_set_interface_alt_setting failed");
      goto fail;
    }

    /* Set up the transfers */
    for (transfer_id = 0; transfer_id < LIBUVC_NUM_TRANSFER_BUFS; ++transfer_id) {
      transfer = libusb_alloc_transfer(packets_per_transfer);
      strmh->transfers[transfer_id] = transfer;      
      strmh->transfer_bufs[transfer_id] = malloc(total_transfer_size);

      libusb_fill_iso_transfer(
        transfer, strmh->devh->usb_devh, format_desc->parent->bEndpointAddress,
        strmh->transfer_bufs[transfer_id],
        total_transfer_size, packets_per_transfer, _uvc_stream_callback, (void*) strmh, 5000);

      libusb_set_iso_packet_lengths(transfer, endpoint_bytes_per_packet);
    }
  } else {
    for (transfer_id = 0; transfer_id < LIBUVC_NUM_TRANSFER_BUFS;
        ++transfer_id) {
      transfer = libusb_alloc_transfer(0);
      strmh->transfers[transfer_id] = transfer;
      strmh->transfer_bufs[transfer_id] = malloc (
          strmh->cur_ctrl.dwMaxPayloadTransferSize );
      libusb_fill_bulk_transfer ( transfer, strmh->devh->usb_devh,
          format_desc->parent->bEndpointAddress,
          strmh->transfer_bufs[transfer_id],
          strmh->cur_ctrl.dwMaxPayloadTransferSize, _uvc_stream_callback,
          ( void* ) strmh, 5000 );
    }
  }

  strmh->user_cb = cb;
  strmh->user_ptr = user_ptr;

  /* If the user wants it, set up a thread that calls the user's function
   * with the contents of each frame.
   */
  if (cb) {
    pthread_create(&strmh->cb_thread, NULL, _uvc_user_caller, (void*) strmh);
  }

  for (transfer_id = 0; transfer_id < LIBUVC_NUM_TRANSFER_BUFS;
      transfer_id++) {
    ret = (uvc_error_t)libusb_submit_transfer(strmh->transfers[transfer_id]);
    if (ret != UVC_SUCCESS) {
      UVC_DEBUG("libusb_submit_transfer failed: %d",ret);
      break;
    }
  }

  if ( ret != UVC_SUCCESS && transfer_id >= 0 ) {
    for ( ; transfer_id < LIBUVC_NUM_TRANSFER_BUFS; transfer_id++) {
      free ( strmh->transfers[transfer_id]->buffer );
      libusb_free_transfer ( strmh->transfers[transfer_id]);
      strmh->transfers[transfer_id] = 0;
    }
    ret = UVC_SUCCESS;
  }

  UVC_EXIT(ret);
  return ret;
fail:
  strmh->running = 0;
  UVC_EXIT(ret);
  return ret;
}

/** Begin streaming video from the stream into the callback function.
 * @ingroup streaming
 *
 * @deprecated The stream type (bulk vs. isochronous) will be determined by the
 * type of interface associated with the uvc_stream_ctrl_t parameter, regardless
 * of whether the caller requests isochronous streaming. Please switch to
 * uvc_stream_start().
 *
 * @param strmh UVC stream
 * @param cb   User callback function. See {uvc_frame_callback_t} for restrictions.
 */
uvc_error_t uvc_stream_start_iso(
    uvc_stream_handle_t *strmh,
    uvc_frame_callback_t *cb,
    void *user_ptr
) {
  return uvc_stream_start(strmh, cb, user_ptr, 0);
}

/** @internal
 * @brief User callback runner thread
 * @note There should be at most one of these per currently streaming device
 * @param arg Device handle
 */
void *_uvc_user_caller(void *arg) {
  uvc_stream_handle_t *strmh = (uvc_stream_handle_t *) arg;

  uint32_t last_seq = 0;

  do {
    pthread_mutex_lock(&strmh->cb_mutex);

    while (strmh->running && last_seq == strmh->hold_seq) {
      pthread_cond_wait(&strmh->cb_cond, &strmh->cb_mutex);
    }

    if (!strmh->running) {
      pthread_mutex_unlock(&strmh->cb_mutex);
      break;
    }
    
    last_seq = strmh->hold_seq;
    _uvc_populate_frame(strmh);
    
    pthread_mutex_unlock(&strmh->cb_mutex);
    
    strmh->user_cb(&strmh->frame, strmh->user_ptr);
  } while(1);

  return NULL; // return value ignored
}

/** @internal
 * @brief Populate the fields of a frame to be handed to user code
 * must be called with stream cb lock held!
 */
void _uvc_populate_frame(uvc_stream_handle_t *strmh) {
  uvc_frame_t *frame = &strmh->frame;
  uvc_frame_desc_t *frame_desc;

  /** @todo this stuff that hits the main config cache should really happen
   * in start() so that only one thread hits these data. all of this stuff
   * is going to be reopen_on_change anyway
   */

  frame_desc = uvc_find_frame_desc(strmh->devh, strmh->cur_ctrl.bFormatIndex,
				   strmh->cur_ctrl.bFrameIndex);

  frame->frame_format = strmh->frame_format;
  
  frame->width = frame_desc->wWidth;
  frame->height = frame_desc->wHeight;
  
  switch (frame->frame_format) {
  case UVC_FRAME_FORMAT_BGR:
    frame->step = frame->width * 3;
    break;
  case UVC_FRAME_FORMAT_YUYV:
    frame->step = frame->width * 2;
    break;
  case UVC_FRAME_FORMAT_NV12:
    frame->step = frame->width;
    break;
  case UVC_FRAME_FORMAT_P010:
    frame->step = frame->width * 2;
        break;
  case UVC_FRAME_FORMAT_MJPEG:
    frame->step = 0;
    break;
  case UVC_FRAME_FORMAT_H264:
    frame->step = 0;
    break;
  default:
    frame->step = 0;
    break;
  }

  frame->sequence = strmh->hold_seq;
  frame->capture_time_finished = strmh->capture_time_finished;

  /* copy the image data from the hold buffer to the frame (unnecessary extra buf?) */
  if (frame->data_bytes < strmh->hold_bytes) {
    frame->data = realloc(frame->data, strmh->hold_bytes);
  }
  frame->data_bytes = strmh->hold_bytes;
  memcpy(frame->data, strmh->holdbuf, frame->data_bytes);

  if (strmh->meta_hold_bytes > 0)
  {
      if (frame->metadata_bytes < strmh->meta_hold_bytes)
      {
          frame->metadata = realloc(frame->metadata, strmh->meta_hold_bytes);
      }
      frame->metadata_bytes = strmh->meta_hold_bytes;
      memcpy(frame->metadata, strmh->meta_holdbuf, frame->metadata_bytes);
  }
}

/** Poll for a frame
 * @ingroup streaming
 *
 * @param devh UVC device
 * @param[out] frame Location to store pointer to captured frame (NULL on error)
 * @param timeout_us >0: Wait at most N microseconds; 0: Wait indefinitely; -1: return immediately
 */
uvc_error_t uvc_stream_get_frame(uvc_stream_handle_t *strmh,
			  uvc_frame_t **frame,
			  int32_t timeout_us) {
  time_t add_secs;
  time_t add_nsecs;
  struct timespec ts;

  if (!strmh->running)
    return UVC_ERROR_INVALID_PARAM;

  if (strmh->user_cb)
    return UVC_ERROR_CALLBACK_EXISTS;

  pthread_mutex_lock(&strmh->cb_mutex);

  if (strmh->last_polled_seq < strmh->hold_seq) {
    _uvc_populate_frame(strmh);
    *frame = &strmh->frame;
    strmh->last_polled_seq = strmh->hold_seq;
  } else if (timeout_us != -1) {
    if (timeout_us == 0) {
      pthread_cond_wait(&strmh->cb_cond, &strmh->cb_mutex);
    } else {
      add_secs = timeout_us / 1000000;
      add_nsecs = (timeout_us % 1000000) * 1000;
      ts.tv_sec = 0;
      ts.tv_nsec = 0;

#if _POSIX_TIMERS > 0
      clock_gettime(CLOCK_REALTIME, &ts);
#else
      struct timeval tv;
      gettimeofday(&tv, NULL);
      ts.tv_sec = tv.tv_sec;
      ts.tv_nsec = tv.tv_usec * 1000;
#endif

      ts.tv_sec += add_secs;
      ts.tv_nsec += add_nsecs;

      /* pthread_cond_timedwait FAILS with EINVAL if ts.tv_nsec > 1000000000 (1 billion)
       * Since we are just adding values to the timespec, we have to increment the seconds if nanoseconds is greater than 1 billion,
       * and then re-adjust the nanoseconds in the correct range.
       * */
      ts.tv_sec += ts.tv_nsec / 1000000000;
      ts.tv_nsec = ts.tv_nsec % 1000000000;

      int err = pthread_cond_timedwait(&strmh->cb_cond, &strmh->cb_mutex, &ts);

      //TODO: How should we handle EINVAL?
      if (err) {
        *frame = NULL;
        pthread_mutex_unlock(&strmh->cb_mutex);
        return err == ETIMEDOUT ? UVC_ERROR_TIMEOUT : UVC_ERROR_OTHER;
      }
    }
    
    if (strmh->last_polled_seq < strmh->hold_seq) {
      _uvc_populate_frame(strmh);
      *frame = &strmh->frame;
      strmh->last_polled_seq = strmh->hold_seq;
    } else {
      *frame = NULL;
    }
  } else {
    *frame = NULL;
  }

  pthread_mutex_unlock(&strmh->cb_mutex);

  return UVC_SUCCESS;
}

/** @brief Stop streaming video
 * @ingroup streaming
 *
 * Closes all streams, ends threads and cancels pollers
 *
 * @param devh UVC device
 */
void uvc_stop_streaming(uvc_device_handle_t *devh) {
  uvc_stream_handle_t *strmh, *strmh_tmp;

  DL_FOREACH_SAFE(devh->streams, strmh, strmh_tmp) {
    uvc_stream_close(strmh);
  }
}

/** @brief Stop stream.
 * @ingroup streaming
 *
 * Stops stream, ends threads and cancels pollers
 *
 * @param devh UVC device
 */
uvc_error_t uvc_stream_stop(uvc_stream_handle_t *strmh) {
  int i;

  if (!strmh->running)
    return UVC_ERROR_INVALID_PARAM;

  strmh->running = 0;

  pthread_mutex_lock(&strmh->cb_mutex);

  /* Attempt to cancel any running transfers, we can't free them just yet because they aren't
   *   necessarily completed but they will be free'd in _uvc_stream_callback().
   */
  for(i=0; i < LIBUVC_NUM_TRANSFER_BUFS; i++) {
    if(strmh->transfers[i] != NULL)
      libusb_cancel_transfer(strmh->transfers[i]);
  }

  /* Wait for transfers to complete/cancel */
  do {
    for(i=0; i < LIBUVC_NUM_TRANSFER_BUFS; i++) {
      if(strmh->transfers[i] != NULL)
        break;
    }
    if(i == LIBUVC_NUM_TRANSFER_BUFS )
      break;
    pthread_cond_wait(&strmh->cb_cond, &strmh->cb_mutex);
  } while(1);
  // Kick the user thread awake
  pthread_cond_broadcast(&strmh->cb_cond);
  pthread_mutex_unlock(&strmh->cb_mutex);

  /** @todo stop the actual stream, camera side? */

  if (strmh->user_cb) {
    /* wait for the thread to stop (triggered by
     * LIBUSB_TRANSFER_CANCELLED transfer) */
    pthread_join(strmh->cb_thread, NULL);
  }

  return UVC_SUCCESS;
}

/** @brief Close stream.
 * @ingroup streaming
 *
 * Closes stream, frees handle and all streaming resources.
 *
 * @param strmh UVC stream handle
 */
void uvc_stream_close(uvc_stream_handle_t *strmh) {
  if (strmh->running)
    uvc_stream_stop(strmh);

  uvc_release_if(strmh->devh, strmh->stream_if->bInterfaceNumber);

  if (strmh->frame.data)
    free(strmh->frame.data);

  free(strmh->outbuf);
  free(strmh->holdbuf);

  free(strmh->meta_outbuf);
  free(strmh->meta_holdbuf);

  pthread_cond_destroy(&strmh->cb_cond);
  pthread_mutex_destroy(&strmh->cb_mutex);

  DL_DELETE(strmh->devh->streams, strmh);
  free(strmh);
}


#endif // STREAM_C

#define INIT_C
#ifdef INIT_C

/** @internal
 * @brief Event handler thread
 * There's one of these per UVC context.
 * @todo We shouldn't run this if we don't own the USB context
 */
void *_uvc_handle_events(void *arg) {
  uvc_context_t *ctx = (uvc_context_t *) arg;

  while (!ctx->kill_handler_thread)
    libusb_handle_events_completed(ctx->usb_ctx, &ctx->kill_handler_thread);
  return NULL;
}

/** @brief Initializes the UVC context
 * @ingroup init
 *
 * @note If you provide your own USB context, you must handle
 * libusb event processing using a function such as libusb_handle_events.
 *
 * @param[out] pctx The location where the context reference should be stored.
 * @param[in]  usb_ctx Optional USB context to use
 * @return Error opening context or UVC_SUCCESS
 */
uvc_error_t uvc_init(uvc_context_t **pctx, struct libusb_context *usb_ctx) {
  uvc_error_t ret = UVC_SUCCESS;
  uvc_context_t *ctx = calloc(1, sizeof(*ctx));

  if (usb_ctx == NULL) {
    ret = (uvc_error_t)libusb_init(&ctx->usb_ctx);
    ctx->own_usb_ctx = 1;
    if (ret != UVC_SUCCESS) {
      free(ctx);
      ctx = NULL;
    }
  } else {
    ctx->own_usb_ctx = 0;
    ctx->usb_ctx = usb_ctx;
  }

  if (ctx != NULL)
    *pctx = ctx;

  return ret;
}

/**
 * @brief Closes the UVC context, shutting down any active cameras.
 * @ingroup init
 *
 * @note This function invalides any existing references to the context's
 * cameras.
 *
 * If no USB context was provided to #uvc_init, the UVC-specific USB
 * context will be destroyed.
 *
 * @param ctx UVC context to shut down
 */
void uvc_exit(uvc_context_t *ctx) {
  uvc_device_handle_t *devh;

  DL_FOREACH(ctx->open_devices, devh) {
    uvc_close(devh);
  }

  if (ctx->own_usb_ctx)
    libusb_exit(ctx->usb_ctx);

  free(ctx);
}

/**
 * @internal
 * @brief Spawns a handler thread for the context
 * @ingroup init
 *
 * This should be called at the end of a successful uvc_open if no devices
 * are already open (and being handled).
 */
void uvc_start_handler_thread(uvc_context_t *ctx) {
  if (ctx->own_usb_ctx)
    pthread_create(&ctx->handler_thread, NULL, _uvc_handle_events, (void*) ctx);
}

#endif // INIT_C

#define FRAME_C
#ifdef FRAME_C

/** @internal */
uvc_error_t uvc_ensure_frame_size(uvc_frame_t *frame, size_t need_bytes) {
  if (frame->library_owns_data) {
    if (!frame->data || frame->data_bytes != need_bytes) {
      frame->data_bytes = need_bytes;
      frame->data = realloc(frame->data, frame->data_bytes);
    }
    if (!frame->data)
      return UVC_ERROR_NO_MEM;
    return UVC_SUCCESS;
  } else {
    if (!frame->data || frame->data_bytes < need_bytes)
      return UVC_ERROR_NO_MEM;
    return UVC_SUCCESS;
  }
}

/** @brief Allocate a frame structure
 * @ingroup frame
 *
 * @param data_bytes Number of bytes to allocate, or zero
 * @return New frame, or NULL on error
 */
uvc_frame_t *uvc_allocate_frame(size_t data_bytes) {
  uvc_frame_t *frame = malloc(sizeof(*frame));

  if (!frame)
    return NULL;

  memset(frame, 0, sizeof(*frame));

  frame->library_owns_data = 1;

  if (data_bytes > 0) {
    frame->data_bytes = data_bytes;
    frame->data = malloc(data_bytes);

    if (!frame->data) {
      free(frame);
      return NULL;
    }
  }

  return frame;
}

/** @brief Free a frame structure
 * @ingroup frame
 *
 * @param frame Frame to destroy
 */
void uvc_free_frame(uvc_frame_t *frame) {
  if (frame->library_owns_data)
  {
    if (frame->data_bytes > 0)
      free(frame->data);
    if (frame->metadata_bytes > 0)
      free(frame->metadata);
  }

  free(frame);
}

static inline unsigned char sat(int i) {
  return (unsigned char)( i >= 255 ? 255 : (i < 0 ? 0 : i));
}

/** @brief Duplicate a frame, preserving color format
 * @ingroup frame
 *
 * @param in Original frame
 * @param out Duplicate frame
 */
uvc_error_t uvc_duplicate_frame(uvc_frame_t *in, uvc_frame_t *out) {
  if (uvc_ensure_frame_size(out, in->data_bytes) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = in->frame_format;
  out->step = in->step;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  memcpy(out->data, in->data, in->data_bytes);

  if (in->metadata && in->metadata_bytes > 0)
  {
      if (out->metadata_bytes < in->metadata_bytes)
      {
          out->metadata = realloc(out->metadata, in->metadata_bytes);
      }
      out->metadata_bytes = in->metadata_bytes;
      memcpy(out->metadata, in->metadata, in->metadata_bytes);
  }

  return UVC_SUCCESS;
}

#define YUYV2RGB_2(pyuv, prgb) { \
    float r = 1.402f * ((pyuv)[3]-128); \
    float g = -0.34414f * ((pyuv)[1]-128) - 0.71414f * ((pyuv)[3]-128); \
    float b = 1.772f * ((pyuv)[1]-128); \
    (prgb)[0] = sat(pyuv[0] + r); \
    (prgb)[1] = sat(pyuv[0] + g); \
    (prgb)[2] = sat(pyuv[0] + b); \
    (prgb)[3] = sat(pyuv[2] + r); \
    (prgb)[4] = sat(pyuv[2] + g); \
    (prgb)[5] = sat(pyuv[2] + b); \
    }
#define IYUYV2RGB_2(pyuv, prgb) { \
    int r = (22987 * ((pyuv)[3] - 128)) >> 14; \
    int g = (-5636 * ((pyuv)[1] - 128) - 11698 * ((pyuv)[3] - 128)) >> 14; \
    int b = (29049 * ((pyuv)[1] - 128)) >> 14; \
    (prgb)[0] = sat(*(pyuv) + r); \
    (prgb)[1] = sat(*(pyuv) + g); \
    (prgb)[2] = sat(*(pyuv) + b); \
    (prgb)[3] = sat((pyuv)[2] + r); \
    (prgb)[4] = sat((pyuv)[2] + g); \
    (prgb)[5] = sat((pyuv)[2] + b); \
    }
#define IYUYV2RGB_16(pyuv, prgb) IYUYV2RGB_8(pyuv, prgb); IYUYV2RGB_8(pyuv + 16, prgb + 24);
#define IYUYV2RGB_8(pyuv, prgb) IYUYV2RGB_4(pyuv, prgb); IYUYV2RGB_4(pyuv + 8, prgb + 12);
#define IYUYV2RGB_4(pyuv, prgb) IYUYV2RGB_2(pyuv, prgb); IYUYV2RGB_2(pyuv + 4, prgb + 6);

/** @brief Convert a frame from YUYV to RGB
 * @ingroup frame
 *
 * @param in YUYV frame
 * @param out RGB frame
 */
uvc_error_t uvc_yuyv2rgb(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_YUYV)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height * 3) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_RGB;
  out->step = in->width * 3;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *prgb = out->data;
  uint8_t *prgb_end = prgb + out->data_bytes;

  while (prgb < prgb_end) {
    IYUYV2RGB_8(pyuv, prgb);

    prgb += 3 * 8;
    pyuv += 2 * 8;
  }

  return UVC_SUCCESS;
}

#define IYUYV2BGR_2(pyuv, pbgr) { \
    int r = (22987 * ((pyuv)[3] - 128)) >> 14; \
    int g = (-5636 * ((pyuv)[1] - 128) - 11698 * ((pyuv)[3] - 128)) >> 14; \
    int b = (29049 * ((pyuv)[1] - 128)) >> 14; \
    (pbgr)[0] = sat(*(pyuv) + b); \
    (pbgr)[1] = sat(*(pyuv) + g); \
    (pbgr)[2] = sat(*(pyuv) + r); \
    (pbgr)[3] = sat((pyuv)[2] + b); \
    (pbgr)[4] = sat((pyuv)[2] + g); \
    (pbgr)[5] = sat((pyuv)[2] + r); \
    }
#define IYUYV2BGR_16(pyuv, pbgr) IYUYV2BGR_8(pyuv, pbgr); IYUYV2BGR_8(pyuv + 16, pbgr + 24);
#define IYUYV2BGR_8(pyuv, pbgr) IYUYV2BGR_4(pyuv, pbgr); IYUYV2BGR_4(pyuv + 8, pbgr + 12);
#define IYUYV2BGR_4(pyuv, pbgr) IYUYV2BGR_2(pyuv, pbgr); IYUYV2BGR_2(pyuv + 4, pbgr + 6);

/** @brief Convert a frame from YUYV to BGR
 * @ingroup frame
 *
 * @param in YUYV frame
 * @param out BGR frame
 */
uvc_error_t uvc_yuyv2bgr(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_YUYV)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height * 3) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_BGR;
  out->step = in->width * 3;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *pbgr = out->data;
  uint8_t *pbgr_end = pbgr + out->data_bytes;

  while (pbgr < pbgr_end) {
    IYUYV2BGR_8(pyuv, pbgr);

    pbgr += 3 * 8;
    pyuv += 2 * 8;
  }

  return UVC_SUCCESS;
}

#define IYUYV2Y(pyuv, py) { \
    (py)[0] = (pyuv[0]); \
    }

/** @brief Convert a frame from YUYV to Y (GRAY8)
 * @ingroup frame
 *
 * @param in YUYV frame
 * @param out GRAY8 frame
 */
uvc_error_t uvc_yuyv2y(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_YUYV)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_GRAY8;
  out->step = in->width;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *py = out->data;
  uint8_t *py_end = py + out->data_bytes;

  while (py < py_end) {
    IYUYV2Y(pyuv, py);

    py += 1;
    pyuv += 2;
  }

  return UVC_SUCCESS;
}

#define IYUYV2UV(pyuv, puv) { \
    (puv)[0] = (pyuv[1]); \
    }

/** @brief Convert a frame from YUYV to UV (GRAY8)
 * @ingroup frame
 *
 * @param in YUYV frame
 * @param out GRAY8 frame
 */
uvc_error_t uvc_yuyv2uv(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_YUYV)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_GRAY8;
  out->step = in->width;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *puv = out->data;
  uint8_t *puv_end = puv + out->data_bytes;

  while (puv < puv_end) {
    IYUYV2UV(pyuv, puv);

    puv += 1;
    pyuv += 2;
  }

  return UVC_SUCCESS;
}

#define IUYVY2RGB_2(pyuv, prgb) { \
    int r = (22987 * ((pyuv)[2] - 128)) >> 14; \
    int g = (-5636 * ((pyuv)[0] - 128) - 11698 * ((pyuv)[2] - 128)) >> 14; \
    int b = (29049 * ((pyuv)[0] - 128)) >> 14; \
    (prgb)[0] = sat((pyuv)[1] + r); \
    (prgb)[1] = sat((pyuv)[1] + g); \
    (prgb)[2] = sat((pyuv)[1] + b); \
    (prgb)[3] = sat((pyuv)[3] + r); \
    (prgb)[4] = sat((pyuv)[3] + g); \
    (prgb)[5] = sat((pyuv)[3] + b); \
    }
#define IUYVY2RGB_16(pyuv, prgb) IUYVY2RGB_8(pyuv, prgb); IUYVY2RGB_8(pyuv + 16, prgb + 24);
#define IUYVY2RGB_8(pyuv, prgb) IUYVY2RGB_4(pyuv, prgb); IUYVY2RGB_4(pyuv + 8, prgb + 12);
#define IUYVY2RGB_4(pyuv, prgb) IUYVY2RGB_2(pyuv, prgb); IUYVY2RGB_2(pyuv + 4, prgb + 6);

/** @brief Convert a frame from UYVY to RGB
 * @ingroup frame
 * @param ini UYVY frame
 * @param out RGB frame
 */
uvc_error_t uvc_uyvy2rgb(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_UYVY)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height * 3) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_RGB;
  out->step = in->width *3;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *prgb = out->data;
  uint8_t *prgb_end = prgb + out->data_bytes;

  while (prgb < prgb_end) {
    IUYVY2RGB_8(pyuv, prgb);

    prgb += 3 * 8;
    pyuv += 2 * 8;
  }

  return UVC_SUCCESS;
}

#define IUYVY2BGR_2(pyuv, pbgr) { \
    int r = (22987 * ((pyuv)[2] - 128)) >> 14; \
    int g = (-5636 * ((pyuv)[0] - 128) - 11698 * ((pyuv)[2] - 128)) >> 14; \
    int b = (29049 * ((pyuv)[0] - 128)) >> 14; \
    (pbgr)[0] = sat((pyuv)[1] + b); \
    (pbgr)[1] = sat((pyuv)[1] + g); \
    (pbgr)[2] = sat((pyuv)[1] + r); \
    (pbgr)[3] = sat((pyuv)[3] + b); \
    (pbgr)[4] = sat((pyuv)[3] + g); \
    (pbgr)[5] = sat((pyuv)[3] + r); \
    }
#define IUYVY2BGR_16(pyuv, pbgr) IUYVY2BGR_8(pyuv, pbgr); IUYVY2BGR_8(pyuv + 16, pbgr + 24);
#define IUYVY2BGR_8(pyuv, pbgr) IUYVY2BGR_4(pyuv, pbgr); IUYVY2BGR_4(pyuv + 8, pbgr + 12);
#define IUYVY2BGR_4(pyuv, pbgr) IUYVY2BGR_2(pyuv, pbgr); IUYVY2BGR_2(pyuv + 4, pbgr + 6);

/** @brief Convert a frame from UYVY to BGR
 * @ingroup frame
 * @param ini UYVY frame
 * @param out BGR frame
 */
uvc_error_t uvc_uyvy2bgr(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_UYVY)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height * 3) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_BGR;
  out->step = in->width *3;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  uint8_t *pyuv = in->data;
  uint8_t *pbgr = out->data;
  uint8_t *pbgr_end = pbgr + out->data_bytes;

  while (pbgr < pbgr_end) {
    IUYVY2BGR_8(pyuv, pbgr);

    pbgr += 3 * 8;
    pyuv += 2 * 8;
  }

  return UVC_SUCCESS;
}

/** @brief Convert a frame to RGB
 * @ingroup frame
 *
 * @param in non-RGB frame
 * @param out RGB frame
 */
uvc_error_t uvc_any2rgb(uvc_frame_t *in, uvc_frame_t *out) {
  switch (in->frame_format) {
#ifdef LIBUVC_HAS_JPEG
    case UVC_FRAME_FORMAT_MJPEG:
      return uvc_mjpeg2rgb(in, out);
#endif
    case UVC_FRAME_FORMAT_YUYV:
      return uvc_yuyv2rgb(in, out);
    case UVC_FRAME_FORMAT_UYVY:
      return uvc_uyvy2rgb(in, out);
    case UVC_FRAME_FORMAT_RGB:
      return uvc_duplicate_frame(in, out);
    default:
      return UVC_ERROR_NOT_SUPPORTED;
  }
}

/** @brief Convert a frame to BGR
 * @ingroup frame
 *
 * @param in non-BGR frame
 * @param out BGR frame
 */
uvc_error_t uvc_any2bgr(uvc_frame_t *in, uvc_frame_t *out) {
  switch (in->frame_format) {
    case UVC_FRAME_FORMAT_YUYV:
      return uvc_yuyv2bgr(in, out);
    case UVC_FRAME_FORMAT_UYVY:
      return uvc_uyvy2bgr(in, out);
    case UVC_FRAME_FORMAT_BGR:
      return uvc_duplicate_frame(in, out);
    default:
      return UVC_ERROR_NOT_SUPPORTED;
  }
}

#endif // FRAME_C

#define FRAME_MJPEG_C
#ifdef FRAME_MJPEG_C

extern uvc_error_t uvc_ensure_frame_size(uvc_frame_t *frame, size_t need_bytes);

struct error_mgr {
  struct jpeg_error_mgr super;
  jmp_buf jmp;
};

static void _error_exit(j_common_ptr dinfo) {
  struct error_mgr *myerr = (struct error_mgr *)dinfo->err;
  (*dinfo->err->output_message)(dinfo);
  longjmp(myerr->jmp, 1);
}

/* ISO/IEC 10918-1:1993(E) K.3.3. Default Huffman tables used by MJPEG UVC devices
   which don't specify a Huffman table in the JPEG stream. */
static const unsigned char dc_lumi_len[] = 
  {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const unsigned char dc_lumi_val[] = 
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const unsigned char dc_chromi_len[] = 
  {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
static const unsigned char dc_chromi_val[] = 
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const unsigned char ac_lumi_len[] = 
  {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
static const unsigned char ac_lumi_val[] = 
  {0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
   0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
   0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1,
   0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
   0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
   0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
   0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
   0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
   0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
   0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
   0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93,
   0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
   0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
   0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
   0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
   0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
   0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
   0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
static const unsigned char ac_chromi_len[] = 
  {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
static const unsigned char ac_chromi_val[] = 
  {0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
   0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
   0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1,
   0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
   0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
   0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36,
   0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47,
   0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
   0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
   0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
   0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a,
   0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
   0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa,
   0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
   0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
   0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
   0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
   0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

#define COPY_HUFF_TABLE(dinfo,tbl,name) do { \
  if(dinfo->tbl == NULL) dinfo->tbl = jpeg_alloc_huff_table((j_common_ptr)dinfo); \
  memcpy(dinfo->tbl->bits, name##_len, sizeof(name##_len)); \
  memset(dinfo->tbl->huffval, 0, sizeof(dinfo->tbl->huffval)); \
  memcpy(dinfo->tbl->huffval, name##_val, sizeof(name##_val)); \
} while(0)

static void insert_huff_tables(j_decompress_ptr dinfo) {
  COPY_HUFF_TABLE(dinfo, dc_huff_tbl_ptrs[0], dc_lumi);
  COPY_HUFF_TABLE(dinfo, dc_huff_tbl_ptrs[1], dc_chromi);
  COPY_HUFF_TABLE(dinfo, ac_huff_tbl_ptrs[0], ac_lumi);
  COPY_HUFF_TABLE(dinfo, ac_huff_tbl_ptrs[1], ac_chromi);
}

static uvc_error_t uvc_mjpeg_convert(uvc_frame_t *in, uvc_frame_t *out) {
  struct jpeg_decompress_struct dinfo;
  struct error_mgr jerr;
  size_t lines_read;
  dinfo.err = jpeg_std_error(&jerr.super);
  jerr.super.error_exit = _error_exit;

  if (setjmp(jerr.jmp)) {
    goto fail;
  }

  jpeg_create_decompress(&dinfo);
  jpeg_mem_src(&dinfo, in->data, in->data_bytes);
  jpeg_read_header(&dinfo, TRUE);

  if (dinfo.dc_huff_tbl_ptrs[0] == NULL) {
    /* This frame is missing the Huffman tables: fill in the standard ones */
    insert_huff_tables(&dinfo);
  }

  if (out->frame_format == UVC_FRAME_FORMAT_RGB)
    dinfo.out_color_space = JCS_RGB;
  else if (out->frame_format == UVC_FRAME_FORMAT_GRAY8)
    dinfo.out_color_space = JCS_GRAYSCALE;
  else
    goto fail;

  dinfo.dct_method = JDCT_IFAST;

  jpeg_start_decompress(&dinfo);

  lines_read = 0;
  while (dinfo.output_scanline < dinfo.output_height) {
    unsigned char *buffer[1] = {( unsigned char*) out->data + lines_read * out->step };
    int num_scanlines;

    num_scanlines = jpeg_read_scanlines(&dinfo, buffer, 1);
    lines_read += num_scanlines;
  }

  jpeg_finish_decompress(&dinfo);
  jpeg_destroy_decompress(&dinfo);
  return 0;

fail:
  jpeg_destroy_decompress(&dinfo);
  return UVC_ERROR_OTHER;
}

/** @brief Convert an MJPEG frame to RGB
 * @ingroup frame
 *
 * @param in MJPEG frame
 * @param out RGB frame
 */
uvc_error_t uvc_mjpeg2rgb(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_MJPEG)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height * 3) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_RGB;
  out->step = in->width * 3;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  return uvc_mjpeg_convert(in, out);
}

/** @brief Convert an MJPEG frame to GRAY8
 * @ingroup frame
 *
 * @param in MJPEG frame
 * @param out GRAY8 frame
 */
uvc_error_t uvc_mjpeg2gray(uvc_frame_t *in, uvc_frame_t *out) {
  if (in->frame_format != UVC_FRAME_FORMAT_MJPEG)
    return UVC_ERROR_INVALID_PARAM;

  if (uvc_ensure_frame_size(out, in->width * in->height) < 0)
    return UVC_ERROR_NO_MEM;

  out->width = in->width;
  out->height = in->height;
  out->frame_format = UVC_FRAME_FORMAT_GRAY8;
  out->step = in->width;
  out->sequence = in->sequence;
  out->capture_time = in->capture_time;
  out->capture_time_finished = in->capture_time_finished;
  out->source = in->source;

  return uvc_mjpeg_convert(in, out);
}

#endif // FRAME_MJPEC_C

#define DIAG_C
#ifdef DIAG_C

/** @internal */
typedef struct _uvc_error_msg {
  uvc_error_t err;
  const char *msg;
} _uvc_error_msg_t;

static const _uvc_error_msg_t uvc_error_msgs[] = {
  {UVC_SUCCESS, "Success"},
  {UVC_ERROR_IO, "I/O error"},
  {UVC_ERROR_INVALID_PARAM, "Invalid parameter"},
  {UVC_ERROR_ACCESS, "Access denied"},
  {UVC_ERROR_NO_DEVICE, "No such device"},
  {UVC_ERROR_NOT_FOUND, "Not found"},
  {UVC_ERROR_BUSY, "Busy"},
  {UVC_ERROR_TIMEOUT, "Timeout"},
  {UVC_ERROR_OVERFLOW, "Overflow"},
  {UVC_ERROR_PIPE, "Pipe"},
  {UVC_ERROR_INTERRUPTED, "Interrupted"},
  {UVC_ERROR_NO_MEM, "Out of memory"},
  {UVC_ERROR_NOT_SUPPORTED, "Not supported"},
  {UVC_ERROR_INVALID_DEVICE, "Invalid device"},
  {UVC_ERROR_INVALID_MODE, "Invalid mode"},
  {UVC_ERROR_CALLBACK_EXISTS, "Callback exists"}
};

/** @brief Print a message explaining an error in the UVC driver
 * @ingroup diag
 *
 * @param err UVC error code
 * @param msg Optional custom message, prepended to output
 */
void uvc_perror(uvc_error_t err, const char *msg) {
  if (msg && *msg) {
    fputs(msg, stderr);
    fputs(": ", stderr);
  }

  fprintf(stderr, "%s (%d)\n", uvc_strerror(err), err);
}

/** @brief Return a string explaining an error in the UVC driver
 * @ingroup diag
 *
 * @param err UVC error code
 * @return error message
 */
const char* uvc_strerror(uvc_error_t err) {
  size_t idx;

  for (idx = 0; idx < sizeof(uvc_error_msgs) / sizeof(*uvc_error_msgs); ++idx) {
    if (uvc_error_msgs[idx].err == err) {
      return uvc_error_msgs[idx].msg;
    }
  }

  return "Unknown error";
}

/** @brief Print the values in a stream control block
 * @ingroup diag
 *
 * @param devh UVC device
 * @param stream Output stream (stderr if NULL)
 */
void uvc_print_stream_ctrl(uvc_stream_ctrl_t *ctrl, FILE *stream) {
  if (stream == NULL)
    stream = stderr;

  fprintf(stream, "bmHint: %04x\n", ctrl->bmHint);
  fprintf(stream, "bFormatIndex: %d\n", ctrl->bFormatIndex);
  fprintf(stream, "bFrameIndex: %d\n", ctrl->bFrameIndex);
  fprintf(stream, "dwFrameInterval: %u\n", ctrl->dwFrameInterval);
  fprintf(stream, "wKeyFrameRate: %d\n", ctrl->wKeyFrameRate);
  fprintf(stream, "wPFrameRate: %d\n", ctrl->wPFrameRate);
  fprintf(stream, "wCompQuality: %d\n", ctrl->wCompQuality);
  fprintf(stream, "wCompWindowSize: %d\n", ctrl->wCompWindowSize);
  fprintf(stream, "wDelay: %d\n", ctrl->wDelay);
  fprintf(stream, "dwMaxVideoFrameSize: %u\n", ctrl->dwMaxVideoFrameSize);
  fprintf(stream, "dwMaxPayloadTransferSize: %u\n", ctrl->dwMaxPayloadTransferSize);
  fprintf(stream, "bInterfaceNumber: %d\n", ctrl->bInterfaceNumber);
}

static const char *_uvc_name_for_format_subtype(uint8_t subtype) {
  switch (subtype) {
  case UVC_VS_FORMAT_UNCOMPRESSED:
    return "UncompressedFormat";
  case UVC_VS_FORMAT_MJPEG:
    return "MJPEGFormat";
  case UVC_VS_FORMAT_FRAME_BASED:
    return "FrameFormat";
  default:
    return "Unknown";
  }
}

/** @brief Print camera capabilities and configuration.
 * @ingroup diag
 *
 * @param devh UVC device
 * @param stream Output stream (stderr if NULL)
 */
void uvc_print_diag(uvc_device_handle_t *devh, FILE *stream) {
  if (stream == NULL)
    stream = stderr;

  if (devh->info->ctrl_if.bcdUVC) {
    uvc_streaming_interface_t *stream_if;
    int stream_idx = 0;

    uvc_device_descriptor_t *desc;
    uvc_get_device_descriptor(devh->dev, &desc);

    fprintf(stream, "DEVICE CONFIGURATION (%04x:%04x/%s) ---\n",
        desc->idVendor, desc->idProduct,
        desc->serialNumber ? desc->serialNumber : "[none]");

    uvc_free_device_descriptor(desc);

    fprintf(stream, "Status: %s\n", devh->streams ? "streaming" : "idle");

    fprintf(stream, "VideoControl:\n"
        "\tbcdUVC: 0x%04x\n",
        devh->info->ctrl_if.bcdUVC);

    DL_FOREACH(devh->info->stream_ifs, stream_if) {
      uvc_format_desc_t *fmt_desc;

      ++stream_idx;

      fprintf(stream, "VideoStreaming(%d):\n"
          "\tbEndpointAddress: %d\n\tFormats:\n",
          stream_idx, stream_if->bEndpointAddress);

      DL_FOREACH(stream_if->format_descs, fmt_desc) {
        uvc_frame_desc_t *frame_desc;
        int i;

        switch (fmt_desc->bDescriptorSubtype) {
          case UVC_VS_FORMAT_UNCOMPRESSED:
          case UVC_VS_FORMAT_MJPEG:
          case UVC_VS_FORMAT_FRAME_BASED:
            fprintf(stream,
                "\t\%s(%d)\n"
                "\t\t  bits per pixel: %d\n"
                "\t\t  GUID: ",
                _uvc_name_for_format_subtype(fmt_desc->bDescriptorSubtype),
                fmt_desc->bFormatIndex,
                fmt_desc->bBitsPerPixel);

            for (i = 0; i < 16; ++i)
              fprintf(stream, "%02x", fmt_desc->guidFormat[i]);

            fprintf(stream, " (%4s)\n", fmt_desc->fourccFormat );

            fprintf(stream,
                "\t\t  default frame: %d\n"
                "\t\t  aspect ratio: %dx%d\n"
                "\t\t  interlace flags: %02x\n"
                "\t\t  copy protect: %02x\n",
                fmt_desc->bDefaultFrameIndex,
                fmt_desc->bAspectRatioX,
                fmt_desc->bAspectRatioY,
                fmt_desc->bmInterlaceFlags,
                fmt_desc->bCopyProtect);

            DL_FOREACH(fmt_desc->frame_descs, frame_desc) {
              uint32_t *interval_ptr;

              fprintf(stream,
                  "\t\t\tFrameDescriptor(%d)\n"
                  "\t\t\t  capabilities: %02x\n"
                  "\t\t\t  size: %dx%d\n"
                  "\t\t\t  bit rate: %d-%d\n"
                  "\t\t\t  max frame size: %d\n"
                  "\t\t\t  default interval: 1/%d\n",
                  frame_desc->bFrameIndex,
                  frame_desc->bmCapabilities,
                  frame_desc->wWidth,
                  frame_desc->wHeight,
                  frame_desc->dwMinBitRate,
                  frame_desc->dwMaxBitRate,
                  frame_desc->dwMaxVideoFrameBufferSize,
                  10000000 / frame_desc->dwDefaultFrameInterval);
              if (frame_desc->intervals) {
                for (interval_ptr = frame_desc->intervals;
                     *interval_ptr;
                     ++interval_ptr) {
                  fprintf(stream,
                      "\t\t\t  interval[%d]: 1/%d\n",
		      (int) (interval_ptr - frame_desc->intervals),
		      10000000 / *interval_ptr);
                }
              } else {
                fprintf(stream,
                    "\t\t\t  min interval[%d] = 1/%d\n"
                    "\t\t\t  max interval[%d] = 1/%d\n",
                    frame_desc->dwMinFrameInterval,
                    10000000 / frame_desc->dwMinFrameInterval,
                    frame_desc->dwMaxFrameInterval,
                    10000000 / frame_desc->dwMaxFrameInterval);
                if (frame_desc->dwFrameIntervalStep)
                  fprintf(stream,
                      "\t\t\t  interval step[%d] = 1/%d\n",
                      frame_desc->dwFrameIntervalStep,
                      10000000 / frame_desc->dwFrameIntervalStep);
              }
            }
            if(fmt_desc->still_frame_desc)
            {
                uvc_still_frame_desc_t* still_frame_desc;
                DL_FOREACH(fmt_desc->still_frame_desc, still_frame_desc)
                {
                    fprintf(stream,
                        "\t\t\tStillFrameDescriptor\n"
                        "\t\t\t  bEndPointAddress: %02x\n",
                        still_frame_desc->bEndPointAddress);
                    uvc_still_frame_res_t* imageSizePattern;
                    DL_FOREACH(still_frame_desc->imageSizePatterns, imageSizePattern) {
                        fprintf(stream,
                            "\t\t\t  wWidth(%d) = %d\n"
                            "\t\t\t  wHeight(%d) = %d\n",
                            imageSizePattern->bResolutionIndex,
                            imageSizePattern->wWidth,
                            imageSizePattern->bResolutionIndex,
                            imageSizePattern->wHeight);
                    }
                }
            }
            break;
          default:
            fprintf(stream, "\t-UnknownFormat (%d)\n",
                fmt_desc->bDescriptorSubtype );
        }
      }
    }

    fprintf(stream, "END DEVICE CONFIGURATION\n");
  } else {
    fprintf(stream, "uvc_print_diag: Device not configured!\n");
  }
}

/** @brief Print all possible frame configuration.
 * @ingroup diag
 *
 * @param devh UVC device
 * @param stream Output stream (stderr if NULL)
 */
void uvc_print_frameformats(uvc_device_handle_t *devh) {

  if (devh->info->ctrl_if.bcdUVC) {
    uvc_streaming_interface_t *stream_if;
    int stream_idx = 0;
    DL_FOREACH(devh->info->stream_ifs, stream_if) {
      uvc_format_desc_t *fmt_desc;
      ++stream_idx;

      DL_FOREACH(stream_if->format_descs, fmt_desc) {
        uvc_frame_desc_t *frame_desc;
        int i;

        switch (fmt_desc->bDescriptorSubtype) {
          case UVC_VS_FORMAT_UNCOMPRESSED:
          case UVC_VS_FORMAT_MJPEG:
          case UVC_VS_FORMAT_FRAME_BASED:
            printf("         \%s(%d)\n"
                "            bits per pixel: %d\n"
                "            GUID: ",
                _uvc_name_for_format_subtype(fmt_desc->bDescriptorSubtype),
                fmt_desc->bFormatIndex,
                fmt_desc->bBitsPerPixel);

            for (i = 0; i < 16; ++i)
              printf("%02x", fmt_desc->guidFormat[i]);

            printf(" (%4s)\n", fmt_desc->fourccFormat );

            printf("            default frame: %d\n"
                "            aspect ratio: %dx%d\n"
                "            interlace flags: %02x\n"
                "            copy protect: %02x\n",
                fmt_desc->bDefaultFrameIndex,
                fmt_desc->bAspectRatioX,
                fmt_desc->bAspectRatioY,
                fmt_desc->bmInterlaceFlags,
                fmt_desc->bCopyProtect);

            DL_FOREACH(fmt_desc->frame_descs, frame_desc) {
              uint32_t *interval_ptr;

              printf("               FrameDescriptor(%d)\n"
                  "                  capabilities: %02x\n"
                  "                  size: %dx%d\n"
                  "                  bit rate: %d-%d\n"
                  "                  max frame size: %d\n"
                  "                  default interval: 1/%d\n",
                  frame_desc->bFrameIndex,
                  frame_desc->bmCapabilities,
                  frame_desc->wWidth,
                  frame_desc->wHeight,
                  frame_desc->dwMinBitRate,
                  frame_desc->dwMaxBitRate,
                  frame_desc->dwMaxVideoFrameBufferSize,
                  10000000 / frame_desc->dwDefaultFrameInterval);
              if (frame_desc->intervals) {
                for (interval_ptr = frame_desc->intervals;
                     *interval_ptr;
                     ++interval_ptr) {
                  printf("                  interval[%d]: 1/%d\n",
		      (int) (interval_ptr - frame_desc->intervals),
		      10000000 / *interval_ptr);
                }
              } else {
                printf("                  min interval[%d] = 1/%d\n"
                    "                  max interval[%d] = 1/%d\n",
                    frame_desc->dwMinFrameInterval,
                    10000000 / frame_desc->dwMinFrameInterval,
                    frame_desc->dwMaxFrameInterval,
                    10000000 / frame_desc->dwMaxFrameInterval);
                if (frame_desc->dwFrameIntervalStep)
                  printf("                  interval step[%d] = 1/%d\n",
                      frame_desc->dwFrameIntervalStep,
                      10000000 / frame_desc->dwFrameIntervalStep);
              }
            }
            break;
          default:
            printf("\t-UnknownFormat (%d)\n",fmt_desc->bDescriptorSubtype );
        }
      }
    }
  } else {
    printf("uvc_print_frameformats: Device not configured!\n");
  }
}

#endif // DIAG_C

#define DEVICE_C
#ifdef DEVICE_C

int uvc_already_open(uvc_context_t *ctx, struct libusb_device *usb_dev);
void uvc_free_devh(uvc_device_handle_t *devh);

uvc_error_t uvc_get_device_info(uvc_device_handle_t *devh, uvc_device_info_t **info);
void uvc_free_device_info(uvc_device_info_t *info);

uvc_error_t uvc_scan_control(uvc_device_handle_t *devh, uvc_device_info_t *info);
uvc_error_t uvc_parse_vc(uvc_device_t *dev,
			 uvc_device_info_t *info,
			 const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vc_selector_unit(uvc_device_t *dev,
					uvc_device_info_t *info,
					const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vc_extension_unit(uvc_device_t *dev,
					uvc_device_info_t *info,
					const unsigned char *block,
					size_t block_size);
uvc_error_t uvc_parse_vc_header(uvc_device_t *dev,
				uvc_device_info_t *info,
				const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vc_input_terminal(uvc_device_t *dev,
					uvc_device_info_t *info,
					const unsigned char *block,
					size_t block_size);
uvc_error_t uvc_parse_vc_processing_unit(uvc_device_t *dev,
					 uvc_device_info_t *info,
					 const unsigned char *block,
					 size_t block_size);

uvc_error_t uvc_scan_streaming(uvc_device_t *dev,
			       uvc_device_info_t *info,
			       int interface_idx);
uvc_error_t uvc_parse_vs(uvc_device_t *dev,
			 uvc_device_info_t *info,
			 uvc_streaming_interface_t *stream_if,
			 const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vs_format_uncompressed(uvc_streaming_interface_t *stream_if,
					     const unsigned char *block,
					     size_t block_size);
uvc_error_t uvc_parse_vs_format_mjpeg(uvc_streaming_interface_t *stream_if,
					     const unsigned char *block,
					     size_t block_size);
uvc_error_t uvc_parse_vs_frame_uncompressed(uvc_streaming_interface_t *stream_if,
					    const unsigned char *block,
					    size_t block_size);
uvc_error_t uvc_parse_vs_frame_format(uvc_streaming_interface_t *stream_if,
					    const unsigned char *block,
					    size_t block_size);
uvc_error_t uvc_parse_vs_frame_frame(uvc_streaming_interface_t *stream_if,
					    const unsigned char *block,
					    size_t block_size);
uvc_error_t uvc_parse_vs_input_header(uvc_streaming_interface_t *stream_if,
				      const unsigned char *block,
				      size_t block_size);

void LIBUSB_CALL _uvc_status_callback(struct libusb_transfer *transfer);

/** @internal
 * @brief Test whether the specified USB device has been opened as a UVC device
 * @ingroup device
 *
 * @param ctx Context in which to search for the UVC device
 * @param usb_dev USB device to find
 * @return true if the device is open in this context
 */
int uvc_already_open(uvc_context_t *ctx, struct libusb_device *usb_dev) {
  uvc_device_handle_t *devh;

  DL_FOREACH(ctx->open_devices, devh) {
    if (usb_dev == devh->dev->usb_dev)
      return 1;
  }

  return 0;
}

/** @brief Finds a camera identified by vendor, product and/or serial number
 * @ingroup device
 *
 * @param[in] ctx UVC context in which to search for the camera
 * @param[out] dev Reference to the camera, or NULL if not found
 * @param[in] vid Vendor ID number, optional
 * @param[in] pid Product ID number, optional
 * @param[in] sn Serial number or NULL
 * @return Error finding device or UVC_SUCCESS
 */
uvc_error_t uvc_find_device(
    uvc_context_t *ctx, uvc_device_t **dev,
    int vid, int pid, const char *sn) {
  uvc_error_t ret = UVC_SUCCESS;

  uvc_device_t **list;
  uvc_device_t *test_dev;
  int dev_idx;
  int found_dev;

  UVC_ENTER();

  ret = uvc_get_device_list(ctx, &list);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  dev_idx = 0;
  found_dev = 0;

  while (!found_dev && (test_dev = list[dev_idx++]) != NULL) {
    uvc_device_descriptor_t *desc;

    if (uvc_get_device_descriptor(test_dev, &desc) != UVC_SUCCESS)
      continue;

    if ((!vid || desc->idVendor == vid)
        && (!pid || desc->idProduct == pid)
        && (!sn || (desc->serialNumber && !strcmp(desc->serialNumber, sn))))
      found_dev = 1;

    uvc_free_device_descriptor(desc);
  }

  if (found_dev)
    uvc_ref_device(test_dev);

  uvc_free_device_list(list, 1);

  if (found_dev) {
    *dev = test_dev;
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS;
  } else {
    UVC_EXIT(UVC_ERROR_NO_DEVICE);
    return UVC_ERROR_NO_DEVICE;
  }
}

/** @brief Finds all cameras identified by vendor, product and/or serial number
 * @ingroup device
 *
 * @param[in] ctx UVC context in which to search for the camera
 * @param[out] devs List of matching cameras
 * @param[in] vid Vendor ID number, optional
 * @param[in] pid Product ID number, optional
 * @param[in] sn Serial number or NULL
 * @return Error finding device or UVC_SUCCESS
 */
uvc_error_t uvc_find_devices(
    uvc_context_t *ctx, uvc_device_t ***devs,
    int vid, int pid, const char *sn) {
  uvc_error_t ret = UVC_SUCCESS;

  uvc_device_t **list;
  uvc_device_t *test_dev;
  int dev_idx;
  int found_dev;

  uvc_device_t **list_internal;
  int num_uvc_devices;

  UVC_ENTER();

  ret = uvc_get_device_list(ctx, &list);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  num_uvc_devices = 0;
  dev_idx = 0;
  found_dev = 0;

  list_internal = malloc(sizeof(*list_internal));
  *list_internal = NULL;

  while ((test_dev = list[dev_idx++]) != NULL) {
    uvc_device_descriptor_t *desc;

    if (uvc_get_device_descriptor(test_dev, &desc) != UVC_SUCCESS)
      continue;

    if ((!vid || desc->idVendor == vid)
        && (!pid || desc->idProduct == pid)
        && (!sn || (desc->serialNumber && !strcmp(desc->serialNumber, sn)))) {
      found_dev = 1;
      uvc_ref_device(test_dev);

      num_uvc_devices++;
      list_internal = realloc(list_internal, (num_uvc_devices + 1) * sizeof(*list_internal));

      list_internal[num_uvc_devices - 1] = test_dev;
      list_internal[num_uvc_devices] = NULL;
    }

    uvc_free_device_descriptor(desc);
  }

  uvc_free_device_list(list, 1);

  if (found_dev) {
    *devs = list_internal;
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS;
  } else {
    UVC_EXIT(UVC_ERROR_NO_DEVICE);
    return UVC_ERROR_NO_DEVICE;
  }
}

/** @brief Get the number of the bus to which the device is attached
 * @ingroup device
 */
uint8_t uvc_get_bus_number(uvc_device_t *dev) {
  return libusb_get_bus_number(dev->usb_dev);
}

/** @brief Get the number assigned to the device within its bus
 * @ingroup device
 */
uint8_t uvc_get_device_address(uvc_device_t *dev) {
  return libusb_get_device_address(dev->usb_dev);
}

static uvc_error_t uvc_open_internal(uvc_device_t *dev, struct libusb_device_handle *usb_devh, uvc_device_handle_t **devh);

#if LIBUSB_API_VERSION >= 0x01000107
/** @brief Wrap a platform-specific system device handle and obtain a UVC device handle.
 * The handle allows you to use libusb to perform I/O on the device in question.
 *
 * On Linux, the system device handle must be a valid file descriptor opened on the device node.
 *
 * The system device handle must remain open until uvc_close() is called. The system device handle will not be closed by uvc_close().
 * @ingroup device
 *
 * @param sys_dev the platform-specific system device handle
 * @param context UVC context to prepare the device
 * @param[out] devh Handle on opened device
 * @return Error opening device or SUCCESS
 */
uvc_error_t uvc_wrap(
    int sys_dev,
    uvc_context_t *context,
    uvc_device_handle_t **devh) {
  uvc_error_t ret;
  struct libusb_device_handle *usb_devh;

  UVC_ENTER();

  uvc_device_t *dev = NULL;
  int err = libusb_wrap_sys_device(context->usb_ctx, sys_dev, &usb_devh);
  UVC_DEBUG("libusb_wrap_sys_device() = %d", err);
  if (err != LIBUSB_SUCCESS) {
    UVC_EXIT(err);
    return err;
  }

  dev = calloc(1, sizeof(uvc_device_t));
  dev->ctx = context;
  dev->usb_dev = libusb_get_device(usb_devh);

  ret = uvc_open_internal(dev, usb_devh, devh);
  UVC_EXIT(ret);
  return ret;
}
#endif

/** @brief Open a UVC device
 * @ingroup device
 *
 * @param dev Device to open
 * @param[out] devh Handle on opened device
 * @return Error opening device or SUCCESS
 */
uvc_error_t uvc_open(
    uvc_device_t *dev,
    uvc_device_handle_t **devh) {
  uvc_error_t ret;
  struct libusb_device_handle *usb_devh;

  UVC_ENTER();

  ret = (uvc_error_t)libusb_open(dev->usb_dev, &usb_devh);
  UVC_DEBUG("libusb_open() = %d", ret);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  ret = uvc_open_internal(dev, usb_devh, devh);
  UVC_EXIT(ret);
  return ret;
}

static uvc_error_t uvc_open_internal(
    uvc_device_t *dev,
    struct libusb_device_handle *usb_devh,
    uvc_device_handle_t **devh) {
  uvc_error_t ret;
  uvc_device_handle_t *internal_devh;
  struct libusb_device_descriptor desc;

  UVC_ENTER();

  uvc_ref_device(dev);

  internal_devh = calloc(1, sizeof(*internal_devh));
  internal_devh->dev = dev;
  internal_devh->usb_devh = usb_devh;

  ret = uvc_get_device_info(internal_devh, &(internal_devh->info));

  if (ret != UVC_SUCCESS)
    goto fail;

  UVC_DEBUG("claiming control interface %d", internal_devh->info->ctrl_if.bInterfaceNumber);
  ret = uvc_claim_if(internal_devh, internal_devh->info->ctrl_if.bInterfaceNumber);
  if (ret != UVC_SUCCESS)
    goto fail;

  libusb_get_device_descriptor(dev->usb_dev, &desc);
  internal_devh->is_isight = (desc.idVendor == 0x05ac && desc.idProduct == 0x8501);

  if (internal_devh->info->ctrl_if.bEndpointAddress) {
    internal_devh->status_xfer = libusb_alloc_transfer(0);
    if (!internal_devh->status_xfer) {
      ret = UVC_ERROR_NO_MEM;
      goto fail;
    }

    libusb_fill_interrupt_transfer(internal_devh->status_xfer,
                                   usb_devh,
                                   internal_devh->info->ctrl_if.bEndpointAddress,
                                   internal_devh->status_buf,
                                   sizeof(internal_devh->status_buf),
                                   _uvc_status_callback,
                                   internal_devh,
                                   0);
    ret = (uvc_error_t)libusb_submit_transfer(internal_devh->status_xfer);
    UVC_DEBUG("libusb_submit_transfer() = %d", ret);

    if (ret) {
      fprintf(stderr,
              "uvc: device has a status interrupt endpoint, but unable to read from it\n");
      goto fail;
    }
  }

  if (dev->ctx->own_usb_ctx && dev->ctx->open_devices == NULL) {
    /* Since this is our first device, we need to spawn the event handler thread */
    uvc_start_handler_thread(dev->ctx);
  }

  DL_APPEND(dev->ctx->open_devices, internal_devh);
  *devh = internal_devh;

  UVC_EXIT(ret);

  return ret;

 fail:
  if ( internal_devh->info ) {
    uvc_release_if(internal_devh, internal_devh->info->ctrl_if.bInterfaceNumber);
  }
  libusb_close(usb_devh);
  uvc_unref_device(dev);
  uvc_free_devh(internal_devh);

  UVC_EXIT(ret);

  return ret;
}

/**
 * @internal
 * @brief Parses the complete device descriptor for a device
 * @ingroup device
 * @note Free *info with uvc_free_device_info when you're done
 *
 * @param dev Device to parse descriptor for
 * @param info Where to store a pointer to the new info struct
 */
uvc_error_t uvc_get_device_info(uvc_device_handle_t *devh,
				uvc_device_info_t **info) {
  uvc_error_t ret;
  uvc_device_info_t *internal_info;

  UVC_ENTER();

  internal_info = calloc(1, sizeof(*internal_info));
  if (!internal_info) {
    UVC_EXIT(UVC_ERROR_NO_MEM);
    return UVC_ERROR_NO_MEM;
  }

  if (libusb_get_config_descriptor(devh->dev->usb_dev,
				   0,
				   &(internal_info->config)) != 0) {
    free(internal_info);
    UVC_EXIT(UVC_ERROR_IO);
    return UVC_ERROR_IO;
  }

  ret = uvc_scan_control(devh, internal_info);
  if (ret != UVC_SUCCESS) {
    uvc_free_device_info(internal_info);
    UVC_EXIT(ret);
    return ret;
  }

  *info = internal_info;

  UVC_EXIT(ret);
  return ret;
}

/**
 * @internal
 * @brief Frees the device descriptor for a device
 * @ingroup device
 *
 * @param info Which device info block to free
 */
void uvc_free_device_info(uvc_device_info_t *info) {
  uvc_input_terminal_t *input_term, *input_term_tmp;
  uvc_processing_unit_t *proc_unit, *proc_unit_tmp;
  uvc_extension_unit_t *ext_unit, *ext_unit_tmp;

  uvc_streaming_interface_t *stream_if, *stream_if_tmp;
  uvc_format_desc_t *format, *format_tmp;
  uvc_frame_desc_t *frame, *frame_tmp;
  uvc_still_frame_desc_t *still_frame, *still_frame_tmp;
  uvc_still_frame_res_t *still_res, *still_res_tmp;

  UVC_ENTER();

  DL_FOREACH_SAFE(info->ctrl_if.input_term_descs, input_term, input_term_tmp) {
    DL_DELETE(info->ctrl_if.input_term_descs, input_term);
    free(input_term);
  }

  DL_FOREACH_SAFE(info->ctrl_if.processing_unit_descs, proc_unit, proc_unit_tmp) {
    DL_DELETE(info->ctrl_if.processing_unit_descs, proc_unit);
    free(proc_unit);
  }

  DL_FOREACH_SAFE(info->ctrl_if.extension_unit_descs, ext_unit, ext_unit_tmp) {
    DL_DELETE(info->ctrl_if.extension_unit_descs, ext_unit);
    free(ext_unit);
  }

  DL_FOREACH_SAFE(info->stream_ifs, stream_if, stream_if_tmp) {
    DL_FOREACH_SAFE(stream_if->format_descs, format, format_tmp) {
      DL_FOREACH_SAFE(format->frame_descs, frame, frame_tmp) {
        if (frame->intervals)
          free(frame->intervals);

        DL_DELETE(format->frame_descs, frame);
        free(frame);
      }

      if(format->still_frame_desc) {
        DL_FOREACH_SAFE(format->still_frame_desc, still_frame, still_frame_tmp) {
          DL_FOREACH_SAFE(still_frame->imageSizePatterns, still_res, still_res_tmp) {
            free(still_res);
          }

          if(still_frame->bCompression) {
              free(still_frame->bCompression);
          }
          free(still_frame);
        }
      }

      DL_DELETE(stream_if->format_descs, format);
      free(format);
    }

    DL_DELETE(info->stream_ifs, stream_if);
    free(stream_if);
  }

  if (info->config)
    libusb_free_config_descriptor(info->config);

  free(info);

  UVC_EXIT_VOID();
}

static uvc_error_t get_device_descriptor(
        uvc_device_handle_t *devh,
        uvc_device_descriptor_t **desc) {
  uvc_device_descriptor_t *desc_internal;
  struct libusb_device_descriptor usb_desc;
  struct libusb_device_handle *usb_devh = devh->usb_devh;
  uvc_error_t ret;

  UVC_ENTER();

  ret = (uvc_error_t)libusb_get_device_descriptor(devh->dev->usb_dev, &usb_desc);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  desc_internal = calloc(1, sizeof(*desc_internal));
  desc_internal->idVendor = usb_desc.idVendor;
  desc_internal->idProduct = usb_desc.idProduct;

  unsigned char buf[64];

  int bytes = libusb_get_string_descriptor_ascii(
          usb_devh, usb_desc.iSerialNumber, buf, sizeof(buf));

  if (bytes > 0)
    desc_internal->serialNumber = strdup((const char*) buf);

  bytes = libusb_get_string_descriptor_ascii(
          usb_devh, usb_desc.iManufacturer, buf, sizeof(buf));

  if (bytes > 0)
    desc_internal->manufacturer = strdup((const char*) buf);

  bytes = libusb_get_string_descriptor_ascii(
          usb_devh, usb_desc.iProduct, buf, sizeof(buf));

  if (bytes > 0)
    desc_internal->product = strdup((const char*) buf);

  *desc = desc_internal;

  UVC_EXIT(ret);
  return ret;
}

/**
 * @brief Get a descriptor that contains the general information about
 * a device
 * @ingroup device
 *
 * Free *desc with uvc_free_device_descriptor when you're done.
 *
 * @param dev Device to fetch information about
 * @param[out] desc Descriptor structure
 * @return Error if unable to fetch information, else SUCCESS
 */
uvc_error_t uvc_get_device_descriptor(
    uvc_device_t *dev,
    uvc_device_descriptor_t **desc) {
  uvc_device_descriptor_t *desc_internal;
  struct libusb_device_descriptor usb_desc;
  struct libusb_device_handle *usb_devh;
  uvc_error_t ret;

  UVC_ENTER();

  ret = (uvc_error_t)libusb_get_device_descriptor(dev->usb_dev, &usb_desc);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  desc_internal = calloc(1, sizeof(*desc_internal));
  desc_internal->idVendor = usb_desc.idVendor;
  desc_internal->idProduct = usb_desc.idProduct;

  if (libusb_open(dev->usb_dev, &usb_devh) == 0) {
    unsigned char buf[64];

    int bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iSerialNumber, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->serialNumber = strdup((const char*) buf);

    bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iManufacturer, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->manufacturer = strdup((const char*) buf);

    bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iProduct, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->product = strdup((const char*) buf);

    libusb_close(usb_devh);
  } else {
    UVC_DEBUG("can't open device %04x:%04x, not fetching serial etc.",
	      usb_desc.idVendor, usb_desc.idProduct);
  }

  *desc = desc_internal;

  UVC_EXIT(ret);
  return ret;
}

/**
 * @brief Frees a device descriptor created with uvc_get_device_descriptor
 * @ingroup device
 *
 * @param desc Descriptor to free
 */
void uvc_free_device_descriptor(
    uvc_device_descriptor_t *desc) {
  UVC_ENTER();

  if (desc->serialNumber)
    free((void*) desc->serialNumber);

  if (desc->manufacturer)
    free((void*) desc->manufacturer);

  if (desc->product)
    free((void*) desc->product);

  free(desc);

  UVC_EXIT_VOID();
}

/**
 * @brief Get a list of the UVC devices attached to the system
 * @ingroup device
 *
 * @note Free the list with uvc_free_device_list when you're done.
 *
 * @param ctx UVC context in which to list devices
 * @param list List of uvc_device structures
 * @return Error if unable to list devices, else SUCCESS
 */
uvc_error_t uvc_get_device_list(
    uvc_context_t *ctx,
    uvc_device_t ***list) {
  struct libusb_device **usb_dev_list;
  struct libusb_device *usb_dev;
  int num_usb_devices;

  uvc_device_t **list_internal;
  int num_uvc_devices;

  /* per device */
  int dev_idx;
  struct libusb_config_descriptor *config;
  struct libusb_device_descriptor desc;
  uint8_t got_interface;

  /* per interface */
  int interface_idx;
  const struct libusb_interface *interface;

  /* per altsetting */
  int altsetting_idx;
  const struct libusb_interface_descriptor *if_desc;

  UVC_ENTER();

  num_usb_devices = libusb_get_device_list(ctx->usb_ctx, &usb_dev_list);

  if (num_usb_devices < 0) {
    UVC_EXIT(UVC_ERROR_IO);
    return UVC_ERROR_IO;
  }

  list_internal = malloc(sizeof(*list_internal));
  *list_internal = NULL;

  num_uvc_devices = 0;
  dev_idx = -1;

  while ((usb_dev = usb_dev_list[++dev_idx]) != NULL) {
    got_interface = 0;

    if (libusb_get_config_descriptor(usb_dev, 0, &config) != 0)
      continue;

    if ( libusb_get_device_descriptor ( usb_dev, &desc ) != LIBUSB_SUCCESS )
      continue;

    for (interface_idx = 0;
	 !got_interface && interface_idx < config->bNumInterfaces;
	 ++interface_idx) {
      interface = &config->interface[interface_idx];

      for (altsetting_idx = 0;
	   !got_interface && altsetting_idx < interface->num_altsetting;
	   ++altsetting_idx) {
	if_desc = &interface->altsetting[altsetting_idx];

        // Skip TIS cameras that definitely aren't UVC even though they might
        // look that way

        if ( 0x199e == desc.idVendor && desc.idProduct  >= 0x8201 &&
            desc.idProduct <= 0x8208 ) {
          continue;
        }

        // Special case for Imaging Source cameras
	/* Video, Streaming */
        if ( 0x199e == desc.idVendor && ( 0x8101 == desc.idProduct ||
            0x8102 == desc.idProduct ) &&
            if_desc->bInterfaceClass == 255 &&
            if_desc->bInterfaceSubClass == 2 ) {
	  got_interface = 1;
	}

	/* Video, Streaming */
	if (if_desc->bInterfaceClass == 14 && if_desc->bInterfaceSubClass == 2) {
	  got_interface = 1;
	}
      }
    }

    libusb_free_config_descriptor(config);

    if (got_interface) {
      uvc_device_t *uvc_dev = malloc(sizeof(*uvc_dev));
      uvc_dev->ctx = ctx;
      uvc_dev->ref = 0;
      uvc_dev->usb_dev = usb_dev;
      uvc_ref_device(uvc_dev);

      num_uvc_devices++;
      list_internal = realloc(list_internal, (num_uvc_devices + 1) * sizeof(*list_internal));

      list_internal[num_uvc_devices - 1] = uvc_dev;
      list_internal[num_uvc_devices] = NULL;

      UVC_DEBUG("    UVC: %d", dev_idx);
    } else {
      UVC_DEBUG("non-UVC: %d", dev_idx);
    }
  }

  libusb_free_device_list(usb_dev_list, 1);

  *list = list_internal;

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/**
 * @brief Frees a list of device structures created with uvc_get_device_list.
 * @ingroup device
 *
 * @param list Device list to free
 * @param unref_devices Decrement the reference counter for each device
 * in the list, and destroy any entries that end up with zero references
 */
void uvc_free_device_list(uvc_device_t **list, uint8_t unref_devices) {
  uvc_device_t *dev;
  int dev_idx = 0;

  UVC_ENTER();

  if (unref_devices) {
    while ((dev = list[dev_idx++]) != NULL) {
      uvc_unref_device(dev);
    }
  }

  free(list);

  UVC_EXIT_VOID();
}

/**
 * @brief Get the uvc_device_t corresponding to an open device
 * @ingroup device
 *
 * @note Unref the uvc_device_t when you're done with it
 *
 * @param devh Device handle to an open UVC device
 */
uvc_device_t *uvc_get_device(uvc_device_handle_t *devh) {
  uvc_ref_device(devh->dev);
  return devh->dev;
}

/**
 * @brief Get the underlying libusb device handle for an open device
 * @ingroup device
 *
 * This can be used to access other interfaces on the same device, e.g.
 * a webcam microphone.
 *
 * @note The libusb device handle is only valid while the UVC device is open;
 * it will be invalidated upon calling uvc_close.
 *
 * @param devh UVC device handle to an open device
 */
libusb_device_handle *uvc_get_libusb_handle(uvc_device_handle_t *devh) {
  return devh->usb_devh;
}

/**
 * @brief Get camera terminal descriptor for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list, but iterating through
 * it will make it no longer the camera terminal
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_input_terminal_t *uvc_get_camera_terminal(uvc_device_handle_t *devh) {
  const uvc_input_terminal_t *term = uvc_get_input_terminals(devh);
  while(term != NULL) {
    if (term->wTerminalType == UVC_ITT_CAMERA) {
      break;
    }
    else {
      term = term->next;
    }
  }
  return term;
}


/**
 * @brief Get input terminal descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_input_terminal_t *uvc_get_input_terminals(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.input_term_descs;
}

/**
 * @brief Get output terminal descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_output_terminal_t *uvc_get_output_terminals(uvc_device_handle_t *devh) {
  return NULL; /* @todo */
}

/**
 * @brief Get selector unit descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_selector_unit_t *uvc_get_selector_units(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.selector_unit_descs;
}

/**
 * @brief Get processing unit descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_processing_unit_t *uvc_get_processing_units(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.processing_unit_descs;
}

/**
 * @brief Get extension unit descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_extension_unit_t *uvc_get_extension_units(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.extension_unit_descs;
}

/**
 * @brief Increment the reference count for a device
 * @ingroup device
 *
 * @param dev Device to reference
 */
void uvc_ref_device(uvc_device_t *dev) {
  UVC_ENTER();

  dev->ref++;
  libusb_ref_device(dev->usb_dev);

  UVC_EXIT_VOID();
}

/**
 * @brief Decrement the reference count for a device
 * @ingropu device
 * @note If the count reaches zero, the device will be discarded
 *
 * @param dev Device to unreference
 */
void uvc_unref_device(uvc_device_t *dev) {
  UVC_ENTER();

  libusb_unref_device(dev->usb_dev);
  dev->ref--;

  if (dev->ref == 0)
    free(dev);

  UVC_EXIT_VOID();
}

/** @internal
 * Claim a UVC interface, detaching the kernel driver if necessary.
 * @ingroup device
 *
 * @param devh UVC device handle
 * @param idx UVC interface index
 */
uvc_error_t uvc_claim_if(uvc_device_handle_t *devh, int idx) {
  int ret = UVC_SUCCESS;

  UVC_ENTER();

  if ( devh->claimed & ( 1 << idx )) {
    UVC_DEBUG("attempt to claim already-claimed interface %d\n", idx );
    UVC_EXIT(ret);
    return ret;
  }

  /* Tell libusb to detach any active kernel drivers. libusb will keep track of whether
   * it found a kernel driver for this interface. */
  ret = (uvc_error_t)libusb_detach_kernel_driver(devh->usb_devh, idx);

  if (ret == UVC_SUCCESS || ret == LIBUSB_ERROR_NOT_FOUND || ret == LIBUSB_ERROR_NOT_SUPPORTED) {
    UVC_DEBUG("claiming interface %d", idx);
    if (!( ret = (uvc_error_t)libusb_claim_interface(devh->usb_devh, idx))) {
      devh->claimed |= ( 1 << idx );
    }
  } else {
    UVC_DEBUG("not claiming interface %d: unable to detach kernel driver (%s)",
              idx, uvc_strerror(ret));
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Release a UVC interface.
 * @ingroup device
 *
 * @param devh UVC device handle
 * @param idx UVC interface index
 */
uvc_error_t uvc_release_if(uvc_device_handle_t *devh, int idx) {
  int ret = UVC_SUCCESS;

  UVC_ENTER();
  UVC_DEBUG("releasing interface %d", idx);
  if (!( devh->claimed & ( 1 << idx ))) {
    UVC_DEBUG("attempt to release unclaimed interface %d\n", idx );
    UVC_EXIT(ret);
    return ret;
  }

  /* libusb_release_interface *should* reset the alternate setting to the first available,
     but sometimes (e.g. on Darwin) it doesn't. Thus, we do it explicitly here.
     This is needed to de-initialize certain cameras. */
  libusb_set_interface_alt_setting(devh->usb_devh, idx, 0);
  ret = (uvc_error_t)libusb_release_interface(devh->usb_devh, idx);

  if (UVC_SUCCESS == ret) {
    devh->claimed &= ~( 1 << idx );
    /* Reattach any kernel drivers that were disabled when we claimed this interface */
    ret = (uvc_error_t)libusb_attach_kernel_driver(devh->usb_devh, idx);

    if (ret == UVC_SUCCESS) {
      UVC_DEBUG("reattached kernel driver to interface %d", idx);
    } else if (ret == LIBUSB_ERROR_NOT_FOUND || ret == LIBUSB_ERROR_NOT_SUPPORTED) {
      ret = UVC_SUCCESS;  /* NOT_FOUND and NOT_SUPPORTED are OK: nothing to do */
    } else {
      UVC_DEBUG("error reattaching kernel driver to interface %d: %s",
                idx, uvc_strerror(ret));
    }
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Find a device's VideoControl interface and process its descriptor
 * @ingroup device
 */
uvc_error_t uvc_scan_control(uvc_device_handle_t *devh, uvc_device_info_t *info) {
  const struct libusb_interface_descriptor *if_desc;
  uvc_error_t parse_ret, ret;
  int interface_idx;
  const unsigned char *buffer;
  size_t buffer_left, block_size;

  UVC_ENTER();

  ret = UVC_SUCCESS;
  if_desc = NULL;

  uvc_device_descriptor_t* dev_desc;
  int haveTISCamera = 0;
  get_device_descriptor ( devh, &dev_desc );
  if ( 0x199e == dev_desc->idVendor && ( 0x8101 == dev_desc->idProduct ||
      0x8102 == dev_desc->idProduct )) {
    haveTISCamera = 1;
  }
  uvc_free_device_descriptor ( dev_desc );

  for (interface_idx = 0; interface_idx < info->config->bNumInterfaces; ++interface_idx) {
    if_desc = &info->config->interface[interface_idx].altsetting[0];

    if ( haveTISCamera && if_desc->bInterfaceClass == 255 && if_desc->bInterfaceSubClass == 1) // Video, Control
      break;

    if (if_desc->bInterfaceClass == 14 && if_desc->bInterfaceSubClass == 1) // Video, Control
      break;

    if_desc = NULL;
  }

  if (if_desc == NULL) {
    UVC_EXIT(UVC_ERROR_INVALID_DEVICE);
    return UVC_ERROR_INVALID_DEVICE;
  }

  info->ctrl_if.bInterfaceNumber = interface_idx;
  if (if_desc->bNumEndpoints != 0) {
    info->ctrl_if.bEndpointAddress = if_desc->endpoint[0].bEndpointAddress;
  }

  buffer = if_desc->extra;
  buffer_left = if_desc->extra_length;

  while (buffer_left >= 3) { // parseX needs to see buf[0,2] = length,type
    block_size = buffer[0];
    parse_ret = uvc_parse_vc(devh->dev, info, buffer, block_size);

    if (parse_ret != UVC_SUCCESS) {
      ret = parse_ret;
      break;
    }

    buffer_left -= block_size;
    buffer += block_size;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoControl header.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_header(uvc_device_t *dev,
				uvc_device_info_t *info,
				const unsigned char *block, size_t block_size) {
  size_t i;
  uvc_error_t scan_ret, ret = UVC_SUCCESS;

  UVC_ENTER();

  /*
  int uvc_version;
  uvc_version = (block[4] >> 4) * 1000 + (block[4] & 0x0f) * 100
    + (block[3] >> 4) * 10 + (block[3] & 0x0f);
  */

  info->ctrl_if.bcdUVC = SW_TO_SHORT(&block[3]);

  switch (info->ctrl_if.bcdUVC) {
  case 0x0100:
    info->ctrl_if.dwClockFrequency = DW_TO_INT(block + 7);
    break;
  case 0x010a:
    info->ctrl_if.dwClockFrequency = DW_TO_INT(block + 7);
    break;
  case 0x0110:
    break;
  default:
    UVC_EXIT(UVC_ERROR_NOT_SUPPORTED);
    return UVC_ERROR_NOT_SUPPORTED;
  }

  for (i = 12; i < block_size; ++i) {
    scan_ret = uvc_scan_streaming(dev, info, block[i]);
    if (scan_ret != UVC_SUCCESS) {
      ret = scan_ret;
      break;
    }
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoControl input terminal.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_input_terminal(uvc_device_t *dev,
					uvc_device_info_t *info,
					const unsigned char *block, size_t block_size) {
  uvc_input_terminal_t *term;
  size_t i;

  UVC_ENTER();

  /* only supporting camera-type input terminals */
  if (SW_TO_SHORT(&block[4]) != UVC_ITT_CAMERA) {
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS;
  }

  term = calloc(1, sizeof(*term));

  term->bTerminalID = block[3];
  term->wTerminalType = SW_TO_SHORT(&block[4]);
  term->wObjectiveFocalLengthMin = SW_TO_SHORT(&block[8]);
  term->wObjectiveFocalLengthMax = SW_TO_SHORT(&block[10]);
  term->wOcularFocalLength = SW_TO_SHORT(&block[12]);

  for (i = 14 + block[14]; i >= 15; --i)
    term->bmControls = block[i] + (term->bmControls << 8);

  DL_APPEND(info->ctrl_if.input_term_descs, term);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoControl processing unit.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_processing_unit(uvc_device_t *dev,
					 uvc_device_info_t *info,
					 const unsigned char *block, size_t block_size) {
  uvc_processing_unit_t *unit;
  size_t i;

  UVC_ENTER();

  unit = calloc(1, sizeof(*unit));
  unit->bUnitID = block[3];
  unit->bSourceID = block[4];

  for (i = 7 + block[7]; i >= 8; --i)
    unit->bmControls = block[i] + (unit->bmControls << 8);

  DL_APPEND(info->ctrl_if.processing_unit_descs, unit);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoControl selector unit.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_selector_unit(uvc_device_t *dev,
					 uvc_device_info_t *info,
					 const unsigned char *block, size_t block_size) {
  uvc_selector_unit_t *unit;

  UVC_ENTER();

  unit = calloc(1, sizeof(*unit));
  unit->bUnitID = block[3];

  DL_APPEND(info->ctrl_if.selector_unit_descs, unit);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoControl extension unit.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_extension_unit(uvc_device_t *dev,
					uvc_device_info_t *info,
					const unsigned char *block, size_t block_size) {
  uvc_extension_unit_t *unit = calloc(1, sizeof(*unit));
  const uint8_t *start_of_controls;
  int size_of_controls, num_in_pins;
  int i;

  UVC_ENTER();

  unit->bUnitID = block[3];
  memcpy(unit->guidExtensionCode, &block[4], 16);

  num_in_pins = block[21];
  size_of_controls = block[22 + num_in_pins];
  start_of_controls = &block[23 + num_in_pins];

  for (i = size_of_controls - 1; i >= 0; --i)
    unit->bmControls = start_of_controls[i] + (unit->bmControls << 8);

  DL_APPEND(info->ctrl_if.extension_unit_descs, unit);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * Process a single VideoControl descriptor block
 * @ingroup device
 */
uvc_error_t uvc_parse_vc(
    uvc_device_t *dev,
    uvc_device_info_t *info,
    const unsigned char *block, size_t block_size) {
  int descriptor_subtype;
  uvc_error_t ret = UVC_SUCCESS;

  UVC_ENTER();

  if (block[1] != 36) { // not a CS_INTERFACE descriptor??
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS; // UVC_ERROR_INVALID_DEVICE;
  }

  descriptor_subtype = block[2];

  switch (descriptor_subtype) {
  case UVC_VC_HEADER:
    ret = uvc_parse_vc_header(dev, info, block, block_size);
    break;
  case UVC_VC_INPUT_TERMINAL:
    ret = uvc_parse_vc_input_terminal(dev, info, block, block_size);
    break;
  case UVC_VC_OUTPUT_TERMINAL:
    break;
  case UVC_VC_SELECTOR_UNIT:
    ret = uvc_parse_vc_selector_unit(dev, info, block, block_size);
    break;
  case UVC_VC_PROCESSING_UNIT:
    ret = uvc_parse_vc_processing_unit(dev, info, block, block_size);
    break;
  case UVC_VC_EXTENSION_UNIT:
    ret = uvc_parse_vc_extension_unit(dev, info, block, block_size);
    break;
  default:
    ret = UVC_ERROR_INVALID_DEVICE;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Process a VideoStreaming interface
 * @ingroup device
 */
uvc_error_t uvc_scan_streaming(uvc_device_t *dev,
			       uvc_device_info_t *info,
			       int interface_idx) {
  const struct libusb_interface_descriptor *if_desc;
  const unsigned char *buffer;
  size_t buffer_left, block_size;
  uvc_error_t ret, parse_ret;
  uvc_streaming_interface_t *stream_if;

  UVC_ENTER();

  ret = UVC_SUCCESS;

  if_desc = &(info->config->interface[interface_idx].altsetting[0]);
  buffer = if_desc->extra;
  buffer_left = if_desc->extra_length;

  stream_if = calloc(1, sizeof(*stream_if));
  stream_if->parent = info;
  stream_if->bInterfaceNumber = if_desc->bInterfaceNumber;
  DL_APPEND(info->stream_ifs, stream_if);

  while (buffer_left >= 3) {
    block_size = buffer[0];
    parse_ret = uvc_parse_vs(dev, info, stream_if, buffer, block_size);

    if (parse_ret != UVC_SUCCESS) {
      ret = parse_ret;
      break;
    }

    buffer_left -= block_size;
    buffer += block_size;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoStreaming header block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_input_header(uvc_streaming_interface_t *stream_if,
				      const unsigned char *block,
				      size_t block_size) {
  UVC_ENTER();

  stream_if->bEndpointAddress = block[6] & 0x8f;
  stream_if->bTerminalLink = block[8];
  stream_if->bStillCaptureMethod = block[9];

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_format_uncompressed(uvc_streaming_interface_t *stream_if,
					     const unsigned char *block,
					     size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  //format->bmCapabilities = block[4];
  //format->bmFlags = block[5];
  memcpy(format->guidFormat, &block[5], 16);
  format->bBitsPerPixel = block[21];
  format->bDefaultFrameIndex = block[22];
  format->bAspectRatioX = block[23];
  format->bAspectRatioY = block[24];
  format->bmInterlaceFlags = block[25];
  format->bCopyProtect = block[26];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming frame format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_format(uvc_streaming_interface_t *stream_if,
					     const unsigned char *block,
					     size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  format->bNumFrameDescriptors = block[4];
  memcpy(format->guidFormat, &block[5], 16);
  format->bBitsPerPixel = block[21];
  format->bDefaultFrameIndex = block[22];
  format->bAspectRatioX = block[23];
  format->bAspectRatioY = block[24];
  format->bmInterlaceFlags = block[25];
  format->bCopyProtect = block[26];
  format->bVariableSize = block[27];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming MJPEG format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_format_mjpeg(uvc_streaming_interface_t *stream_if,
					     const unsigned char *block,
					     size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  memcpy(format->fourccFormat, "MJPG", 4);
  format->bmFlags = block[5];
  format->bBitsPerPixel = 0;
  format->bDefaultFrameIndex = block[6];
  format->bAspectRatioX = block[7];
  format->bAspectRatioY = block[8];
  format->bmInterlaceFlags = block[9];
  format->bCopyProtect = block[10];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed frame block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_frame(uvc_streaming_interface_t *stream_if,
					    const unsigned char *block,
					    size_t block_size) {
  uvc_format_desc_t *format;
  uvc_frame_desc_t *frame;

  const unsigned char *p;
  int i;

  UVC_ENTER();

  format = stream_if->format_descs->prev;
  frame = calloc(1, sizeof(*frame));

  frame->parent = format;

  frame->bDescriptorSubtype = block[2];
  frame->bFrameIndex = block[3];
  frame->bmCapabilities = block[4];
  frame->wWidth = block[5] + (block[6] << 8);
  frame->wHeight = block[7] + (block[8] << 8);
  frame->dwMinBitRate = DW_TO_INT(&block[9]);
  frame->dwMaxBitRate = DW_TO_INT(&block[13]);
  frame->dwDefaultFrameInterval = DW_TO_INT(&block[17]);
  frame->bFrameIntervalType = block[21];
  frame->dwBytesPerLine = DW_TO_INT(&block[22]);

  if (block[21] == 0) {
    frame->dwMinFrameInterval = DW_TO_INT(&block[26]);
    frame->dwMaxFrameInterval = DW_TO_INT(&block[30]);
    frame->dwFrameIntervalStep = DW_TO_INT(&block[34]);
  } else {
    frame->intervals = calloc(block[21] + 1, sizeof(frame->intervals[0]));
    p = &block[26];

    for (i = 0; i < block[21]; ++i) {
      frame->intervals[i] = DW_TO_INT(p);
      p += 4;
    }
    frame->intervals[block[21]] = 0;
  }

  DL_APPEND(format->frame_descs, frame);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed frame block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_uncompressed(uvc_streaming_interface_t *stream_if,
					    const unsigned char *block,
					    size_t block_size) {
  uvc_format_desc_t *format;
  uvc_frame_desc_t *frame;

  const unsigned char *p;
  int i;

  UVC_ENTER();

  format = stream_if->format_descs->prev;
  frame = calloc(1, sizeof(*frame));

  frame->parent = format;

  frame->bDescriptorSubtype = block[2];
  frame->bFrameIndex = block[3];
  frame->bmCapabilities = block[4];
  frame->wWidth = block[5] + (block[6] << 8);
  frame->wHeight = block[7] + (block[8] << 8);
  frame->dwMinBitRate = DW_TO_INT(&block[9]);
  frame->dwMaxBitRate = DW_TO_INT(&block[13]);
  frame->dwMaxVideoFrameBufferSize = DW_TO_INT(&block[17]);
  frame->dwDefaultFrameInterval = DW_TO_INT(&block[21]);
  frame->bFrameIntervalType = block[25];

  if (block[25] == 0) {
    frame->dwMinFrameInterval = DW_TO_INT(&block[26]);
    frame->dwMaxFrameInterval = DW_TO_INT(&block[30]);
    frame->dwFrameIntervalStep = DW_TO_INT(&block[34]);
  } else {
    frame->intervals = calloc(block[25] + 1, sizeof(frame->intervals[0]));
    p = &block[26];

    for (i = 0; i < block[25]; ++i) {
      frame->intervals[i] = DW_TO_INT(p);
      p += 4;
    }
    frame->intervals[block[25]] = 0;
  }

  DL_APPEND(format->frame_descs, frame);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming still iamge frame
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_still_image_frame(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size) {

  struct uvc_still_frame_desc* frame;
  uvc_format_desc_t *format;

  const unsigned char *p;
  int i;

  UVC_ENTER();

  format = stream_if->format_descs->prev;
  frame = calloc(1, sizeof(*frame));

  frame->parent = format;

  frame->bDescriptorSubtype = block[2];
  frame->bEndPointAddress   = block[3];
  uint8_t numImageSizePatterns = block[4];

  frame->imageSizePatterns = NULL;

  p = &block[5];

  for (i = 1; i <= numImageSizePatterns; ++i) {
    uvc_still_frame_res_t* res = calloc(1, sizeof(uvc_still_frame_res_t));
    res->bResolutionIndex = i;
    res->wWidth = SW_TO_SHORT(p);
    p += 2;
    res->wHeight = SW_TO_SHORT(p);
    p += 2;

    DL_APPEND(frame->imageSizePatterns, res);
  }

  p = &block[5+4*numImageSizePatterns];
  frame->bNumCompressionPattern = *p;

  if(frame->bNumCompressionPattern)
  {
      frame->bCompression = calloc(frame->bNumCompressionPattern, sizeof(frame->bCompression[0]));
      for(i = 0; i < frame->bNumCompressionPattern; ++i)
      {
          ++p;
          frame->bCompression[i] = *p;
      }
  }
  else
  {
      frame->bCompression = NULL;
  }

  DL_APPEND(format->still_frame_desc, frame);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * Process a single VideoStreaming descriptor block
 * @ingroup device
 */
uvc_error_t uvc_parse_vs(
    uvc_device_t *dev,
    uvc_device_info_t *info,
    uvc_streaming_interface_t *stream_if,
    const unsigned char *block, size_t block_size) {
  uvc_error_t ret;
  int descriptor_subtype;

  UVC_ENTER();

  ret = UVC_SUCCESS;
  descriptor_subtype = block[2];

  switch (descriptor_subtype) {
  case UVC_VS_INPUT_HEADER:
    ret = uvc_parse_vs_input_header(stream_if, block, block_size);
    break;
  case UVC_VS_OUTPUT_HEADER:
    UVC_DEBUG("unsupported descriptor subtype VS_OUTPUT_HEADER");
    break;
  case UVC_VS_STILL_IMAGE_FRAME:
    ret = uvc_parse_vs_still_image_frame(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_UNCOMPRESSED:
    ret = uvc_parse_vs_format_uncompressed(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_MJPEG:
    ret = uvc_parse_vs_format_mjpeg(stream_if, block, block_size);
    break;
  case UVC_VS_FRAME_UNCOMPRESSED:
  case UVC_VS_FRAME_MJPEG:
    ret = uvc_parse_vs_frame_uncompressed(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_MPEG2TS:
    UVC_DEBUG("unsupported descriptor subtype VS_FORMAT_MPEG2TS");
    break;
  case UVC_VS_FORMAT_DV:
    UVC_DEBUG("unsupported descriptor subtype VS_FORMAT_DV");
    break;
  case UVC_VS_COLORFORMAT:
    UVC_DEBUG("unsupported descriptor subtype VS_COLORFORMAT");
    break;
  case UVC_VS_FORMAT_FRAME_BASED:
    ret = uvc_parse_vs_frame_format ( stream_if, block, block_size );
    break;
  case UVC_VS_FRAME_FRAME_BASED:
    ret = uvc_parse_vs_frame_frame ( stream_if, block, block_size );
    break;
  case UVC_VS_FORMAT_STREAM_BASED:
    UVC_DEBUG("unsupported descriptor subtype VS_FORMAT_STREAM_BASED");
    break;
  default:
    /** @todo handle JPEG and maybe still frames or even DV... */
    //UVC_DEBUG("unsupported descriptor subtype: %d",descriptor_subtype);
    break;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Free memory associated with a UVC device
 * @pre Streaming must be stopped, and threads must have died
 */
void uvc_free_devh(uvc_device_handle_t *devh) {
  UVC_ENTER();

  if (devh->info)
    uvc_free_device_info(devh->info);

  if (devh->status_xfer)
    libusb_free_transfer(devh->status_xfer);

  free(devh);

  UVC_EXIT_VOID();
}

/** @brief Close a device
 *
 * @ingroup device
 *
 * Ends any stream that's in progress.
 *
 * The device handle and frame structures will be invalidated.
 */
void uvc_close(uvc_device_handle_t *devh) {
  UVC_ENTER();
  uvc_context_t *ctx = devh->dev->ctx;

  if (devh->streams)
    uvc_stop_streaming(devh);

  uvc_release_if(devh, devh->info->ctrl_if.bInterfaceNumber);

  /* If we are managing the libusb context and this is the last open device,
   * then we need to cancel the handler thread. When we call libusb_close,
   * it'll cause a return from the thread's libusb_handle_events call, after
   * which the handler thread will check the flag we set and then exit. */
  if (ctx->own_usb_ctx && ctx->open_devices == devh && devh->next == NULL) {
    ctx->kill_handler_thread = 1;
    libusb_close(devh->usb_devh);
    pthread_join(ctx->handler_thread, NULL);
  } else {
    libusb_close(devh->usb_devh);
  }

  DL_DELETE(ctx->open_devices, devh);

  uvc_unref_device(devh->dev);

  uvc_free_devh(devh);

  UVC_EXIT_VOID();
}

/** @internal
 * @brief Get number of open devices
 */
size_t uvc_num_devices(uvc_context_t *ctx) {
  size_t count = 0;

  uvc_device_handle_t *devh;

  UVC_ENTER();

  DL_FOREACH(ctx->open_devices, devh) {
    count++;
  }

  UVC_EXIT((int) count);
  return count;
}

void uvc_process_control_status(uvc_device_handle_t *devh, unsigned char *data, int len) {
  enum uvc_status_class status_class;
  uint8_t originator = 0, selector = 0, event = 0;
  enum uvc_status_attribute attribute = UVC_STATUS_ATTRIBUTE_UNKNOWN;
  void *content = NULL;
  size_t content_len = 0;
  int found_entity = 0;
  struct uvc_input_terminal *input_terminal;
  struct uvc_processing_unit *processing_unit;

  UVC_ENTER();

  if (len < 5) {
    UVC_DEBUG("Short read of VideoControl status update (%d bytes)", len);
    UVC_EXIT_VOID();
    return;
  }

  originator = data[1];
  event = data[2];
  selector = data[3];

  if (originator == 0) {
    UVC_DEBUG("Unhandled update from VC interface");
    UVC_EXIT_VOID();
    return;  /* @todo VideoControl virtual entity interface updates */
  }

  if (event != 0) {
    UVC_DEBUG("Unhandled VC event %d", (int) event);
    UVC_EXIT_VOID();
    return;
  }

  /* printf("bSelector: %d\n", selector); */

  DL_FOREACH(devh->info->ctrl_if.input_term_descs, input_terminal) {
    if (input_terminal->bTerminalID == originator) {
      status_class = UVC_STATUS_CLASS_CONTROL_CAMERA;
      found_entity = 1;
      break;
    }
  }

  if (!found_entity) {
    DL_FOREACH(devh->info->ctrl_if.processing_unit_descs, processing_unit) {
      if (processing_unit->bUnitID == originator) {
        status_class = UVC_STATUS_CLASS_CONTROL_PROCESSING;
        found_entity = 1;
        break;
      }
    }
  }

  if (!found_entity) {
    UVC_DEBUG("Got status update for unknown VideoControl entity %d",
  (int) originator);
    UVC_EXIT_VOID();
    return;
  }

  attribute = data[4];
  content = data + 5;
  content_len = len - 5;

  UVC_DEBUG("Event: class=%d, event=%d, selector=%d, attribute=%d, content_len=%zd",
    status_class, event, selector, attribute, content_len);

  if(devh->status_cb) {
    UVC_DEBUG("Running user-supplied status callback");
    devh->status_cb(status_class,
                    event,
                    selector,
                    attribute,
                    content, content_len,
                    devh->status_user_ptr);
  }
  
  UVC_EXIT_VOID();
}

void uvc_process_streaming_status(uvc_device_handle_t *devh, unsigned char *data, int len) {
  
  UVC_ENTER();

  if (len < 3) {
    UVC_DEBUG("Invalid streaming status event received.\n");
    UVC_EXIT_VOID();
    return;
  }

  if (data[2] == 0) {
    if (len < 4) {
      UVC_DEBUG("Short read of status update (%d bytes)", len);
      UVC_EXIT_VOID();
      return;
    }
    UVC_DEBUG("Button (intf %u) %s len %d\n", data[1], data[3] ? "pressed" : "released", len);
    
    if(devh->button_cb) {
      UVC_DEBUG("Running user-supplied button callback");
      devh->button_cb(data[1],
                      data[3],
                      devh->button_user_ptr);
    }
  } else {
    UVC_DEBUG("Stream %u error event %02x %02x len %d.\n", data[1], data[2], data[3], len);
  }

  UVC_EXIT_VOID();
}

void uvc_process_status_xfer(uvc_device_handle_t *devh, struct libusb_transfer *transfer) {
  
  UVC_ENTER();

  /* printf("Got transfer of aLen = %d\n", transfer->actual_length); */

  if (transfer->actual_length > 0) {
    switch (transfer->buffer[0] & 0x0f) {
    case 1: /* VideoControl interface */
      uvc_process_control_status(devh, transfer->buffer, transfer->actual_length);
      break;
    case 2:  /* VideoStreaming interface */
      uvc_process_streaming_status(devh, transfer->buffer, transfer->actual_length);
      break;
    }
  }

  UVC_EXIT_VOID();
}

/** @internal
 * @brief Process asynchronous status updates from the device.
 */
void LIBUSB_CALL _uvc_status_callback(struct libusb_transfer *transfer) {
  UVC_ENTER();

  uvc_device_handle_t *devh = (uvc_device_handle_t *) transfer->user_data;

  switch (transfer->status) {
  case LIBUSB_TRANSFER_ERROR:
  case LIBUSB_TRANSFER_CANCELLED:
  case LIBUSB_TRANSFER_NO_DEVICE:
    UVC_DEBUG("not processing/resubmitting, status = %d", transfer->status);
    UVC_EXIT_VOID();
    return;
  case LIBUSB_TRANSFER_COMPLETED:
    uvc_process_status_xfer(devh, transfer);
    break;
  case LIBUSB_TRANSFER_TIMED_OUT:
  case LIBUSB_TRANSFER_STALL:
  case LIBUSB_TRANSFER_OVERFLOW:
    UVC_DEBUG("retrying transfer, status = %d", transfer->status);
    break;
  }

#ifdef UVC_DEBUGGING
  uvc_error_t ret =
#endif
      libusb_submit_transfer(transfer);
  UVC_DEBUG("libusb_submit_transfer() = %d", ret);

  UVC_EXIT_VOID();
}

/** @brief Set a callback function to receive status updates
 *
 * @ingroup device
 */
void uvc_set_status_callback(uvc_device_handle_t *devh,
                             uvc_status_callback_t cb,
                             void *user_ptr) {
  UVC_ENTER();

  devh->status_cb = cb;
  devh->status_user_ptr = user_ptr;

  UVC_EXIT_VOID();
}

/** @brief Set a callback function to receive button events
 *
 * @ingroup device
 */
void uvc_set_button_callback(uvc_device_handle_t *devh,
                             uvc_button_callback_t cb,
                             void *user_ptr) {
  UVC_ENTER();

  devh->button_cb = cb;
  devh->button_user_ptr = user_ptr;

  UVC_EXIT_VOID();
}

/**
 * @brief Get format descriptions for the open device.
 *
 * @note Do not modify the returned structure.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_format_desc_t *uvc_get_format_descs(uvc_device_handle_t *devh) {
  return devh->info->stream_ifs->format_descs;
}

#endif // DEVICE_C

#define CTRL_C
#ifdef CTRL_C

static const int REQ_TYPE_SET = 0x21;
static const int REQ_TYPE_GET = 0xa1;

/***** GENERIC CONTROLS *****/
/**
 * @brief Get the length of a control on a terminal or unit.
 * 
 * @param devh UVC device handle
 * @param unit Unit or Terminal ID; obtain this from the uvc_extension_unit_t describing the extension unit
 * @param ctrl Vendor-specific control number to query
 * @return On success, the length of the control as reported by the device. Otherwise,
 *   a uvc_error_t error describing the error encountered.
 * @ingroup ctrl
 */
int uvc_get_ctrl_len(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl) {
  unsigned char buf[2];

  int ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, UVC_GET_LEN,
    ctrl << 8,
    unit << 8 | devh->info->ctrl_if.bInterfaceNumber,		// XXX saki
    buf,
    2,
    0 /* timeout */);

  if (ret < 0)
    return ret;
  else
    return (unsigned short)SW_TO_SHORT(buf);
}

/**
 * @brief Perform a GET_* request from an extension unit.
 * 
 * @param devh UVC device handle
 * @param unit Unit ID; obtain this from the uvc_extension_unit_t describing the extension unit
 * @param ctrl Control number to query
 * @param data Data buffer to be filled by the device
 * @param len Size of data buffer
 * @param req_code GET_* request to execute
 * @return On success, the number of bytes actually transferred. Otherwise,
 *   a uvc_error_t error describing the error encountered.
 * @ingroup ctrl
 */
int uvc_get_ctrl(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl, void *data, int len, enum uvc_req_code req_code) {
  return libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    ctrl << 8,
    unit << 8 | devh->info->ctrl_if.bInterfaceNumber,		// XXX saki
    data,
    len,
    0 /* timeout */);
}

/**
 * @brief Perform a SET_CUR request to a terminal or unit.
 * 
 * @param devh UVC device handle
 * @param unit Unit or Terminal ID
 * @param ctrl Control number to set
 * @param data Data buffer to be sent to the device
 * @param len Size of data buffer
 * @return On success, the number of bytes actually transferred. Otherwise,
 *   a uvc_error_t error describing the error encountered.
 * @ingroup ctrl
 */
int uvc_set_ctrl(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl, void *data, int len) {
  return libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    ctrl << 8,
    unit << 8 | devh->info->ctrl_if.bInterfaceNumber,		// XXX saki
    data,
    len,
    0 /* timeout */);
}

/***** INTERFACE CONTROLS *****/
uvc_error_t uvc_get_power_mode(uvc_device_handle_t *devh, enum uvc_device_power_mode *mode, enum uvc_req_code req_code) {
  uint8_t mode_char;
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_VC_VIDEO_POWER_MODE_CONTROL << 8,
    devh->info->ctrl_if.bInterfaceNumber,	// XXX saki
    &mode_char,
    sizeof(mode_char),
    0);

  if (ret == 1) {
    *mode = mode_char;
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}

uvc_error_t uvc_set_power_mode(uvc_device_handle_t *devh, enum uvc_device_power_mode mode) {
  uint8_t mode_char = mode;
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_VC_VIDEO_POWER_MODE_CONTROL << 8,
    devh->info->ctrl_if.bInterfaceNumber,	// XXX saki
    &mode_char,
    sizeof(mode_char),
    0);

  if (ret == 1)
    return UVC_SUCCESS;
  else
    return ret;
}

/** @todo Request Error Code Control (UVC 1.5, 4.2.1.2) */

#endif // CTRL_C

#define CTRL_GEN_C
#ifdef CTRL_GEN_C

static const int REQ_TYPE_SET = 0x21;
static const int REQ_TYPE_GET = 0xa1;

/** @ingroup ctrl
 * @brief Reads the SCANNING_MODE control.
 * @param devh UVC device handle
 * @param[out] mode 0: interlaced, 1: progressive
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_scanning_mode(uvc_device_handle_t *devh, uint8_t* mode, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_SCANNING_MODE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *mode = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the SCANNING_MODE control.
 * @param devh UVC device handle
 * @param mode 0: interlaced, 1: progressive
 */
uvc_error_t uvc_set_scanning_mode(uvc_device_handle_t *devh, uint8_t mode) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = mode;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_SCANNING_MODE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads camera's auto-exposure mode.
 * 
 * See uvc_set_ae_mode() for a description of the available modes.
 * @param devh UVC device handle
 * @param[out] mode 1: manual mode; 2: auto mode; 4: shutter priority mode; 8: aperture priority mode
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_ae_mode(uvc_device_handle_t *devh, uint8_t* mode, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_AE_MODE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *mode = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets camera's auto-exposure mode.
 * 
 * Cameras may support any of the following AE modes:
 *  * UVC_AUTO_EXPOSURE_MODE_MANUAL (1) - manual exposure time, manual iris
 *  * UVC_AUTO_EXPOSURE_MODE_AUTO (2) - auto exposure time, auto iris
 *  * UVC_AUTO_EXPOSURE_MODE_SHUTTER_PRIORITY (4) - manual exposure time, auto iris
 *  * UVC_AUTO_EXPOSURE_MODE_APERTURE_PRIORITY (8) - auto exposure time, manual iris
 * 
 * Most cameras provide manual mode and aperture priority mode.
 * @param devh UVC device handle
 * @param mode 1: manual mode; 2: auto mode; 4: shutter priority mode; 8: aperture priority mode
 */
uvc_error_t uvc_set_ae_mode(uvc_device_handle_t *devh, uint8_t mode) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = mode;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_AE_MODE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Checks whether the camera may vary the frame rate for exposure control reasons.
 * See uvc_set_ae_priority() for a description of the `priority` field.
 * @param devh UVC device handle
 * @param[out] priority 0: frame rate must remain constant; 1: frame rate may be varied for AE purposes
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_ae_priority(uvc_device_handle_t *devh, uint8_t* priority, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_AE_PRIORITY_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *priority = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Chooses whether the camera may vary the frame rate for exposure control reasons.
 * A `priority` value of zero means the camera may not vary its frame rate. A value of 1
 * means the frame rate is variable. This setting has no effect outside of the `auto` and
 * `shutter_priority` auto-exposure modes.
 * @param devh UVC device handle
 * @param priority 0: frame rate must remain constant; 1: frame rate may be varied for AE purposes
 */
uvc_error_t uvc_set_ae_priority(uvc_device_handle_t *devh, uint8_t priority) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = priority;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_AE_PRIORITY_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Gets the absolute exposure time.
 * 
 * See uvc_set_exposure_abs() for a description of the `time` field.
 * @param devh UVC device handle
 * @param[out] time 
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_exposure_abs(uvc_device_handle_t *devh, uint32_t* time, enum uvc_req_code req_code) {
  uint8_t data[4];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *time = DW_TO_INT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the absolute exposure time.
 * 
 * The `time` parameter should be provided in units of 0.0001 seconds (e.g., use the value 100
 * for a 10ms exposure period). Auto exposure should be set to `manual` or `shutter_priority`
 * before attempting to change this setting.
 * @param devh UVC device handle
 * @param time 
 */
uvc_error_t uvc_set_exposure_abs(uvc_device_handle_t *devh, uint32_t time) {
  uint8_t data[4];
  uvc_error_t ret;

  INT_TO_DW(time, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the exposure time relative to the current setting.
 * @param devh UVC device handle
 * @param[out] step number of steps by which to change the exposure time, or zero to set the default exposure time
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_exposure_rel(uvc_device_handle_t *devh, int8_t* step, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *step = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the exposure time relative to the current setting.
 * @param devh UVC device handle
 * @param step number of steps by which to change the exposure time, or zero to set the default exposure time
 */
uvc_error_t uvc_set_exposure_rel(uvc_device_handle_t *devh, int8_t step) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = step;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the distance at which an object is optimally focused.
 * @param devh UVC device handle
 * @param[out] focus focal target distance in millimeters
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_focus_abs(uvc_device_handle_t *devh, uint16_t* focus, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_FOCUS_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *focus = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the distance at which an object is optimally focused.
 * @param devh UVC device handle
 * @param focus focal target distance in millimeters
 */
uvc_error_t uvc_set_focus_abs(uvc_device_handle_t *devh, uint16_t focus) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(focus, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_FOCUS_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the FOCUS_RELATIVE control.
 * @param devh UVC device handle
 * @param[out] focus_rel TODO
 * @param[out] speed TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_focus_rel(uvc_device_handle_t *devh, int8_t* focus_rel, uint8_t* speed, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_FOCUS_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *focus_rel = data[0];
    *speed = data[1];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the FOCUS_RELATIVE control.
 * @param devh UVC device handle
 * @param focus_rel TODO
 * @param speed TODO
 */
uvc_error_t uvc_set_focus_rel(uvc_device_handle_t *devh, int8_t focus_rel, uint8_t speed) {
  uint8_t data[2];
  uvc_error_t ret;

  data[0] = focus_rel;
  data[1] = speed;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_FOCUS_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the FOCUS_SIMPLE control.
 * @param devh UVC device handle
 * @param[out] focus TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_focus_simple_range(uvc_device_handle_t *devh, uint8_t* focus, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_FOCUS_SIMPLE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *focus = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the FOCUS_SIMPLE control.
 * @param devh UVC device handle
 * @param focus TODO
 */
uvc_error_t uvc_set_focus_simple_range(uvc_device_handle_t *devh, uint8_t focus) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = focus;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_FOCUS_SIMPLE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the FOCUS_AUTO control.
 * @param devh UVC device handle
 * @param[out] state TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_focus_auto(uvc_device_handle_t *devh, uint8_t* state, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_FOCUS_AUTO_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *state = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the FOCUS_AUTO control.
 * @param devh UVC device handle
 * @param state TODO
 */
uvc_error_t uvc_set_focus_auto(uvc_device_handle_t *devh, uint8_t state) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = state;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_FOCUS_AUTO_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the IRIS_ABSOLUTE control.
 * @param devh UVC device handle
 * @param[out] iris TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_iris_abs(uvc_device_handle_t *devh, uint16_t* iris, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_IRIS_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *iris = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the IRIS_ABSOLUTE control.
 * @param devh UVC device handle
 * @param iris TODO
 */
uvc_error_t uvc_set_iris_abs(uvc_device_handle_t *devh, uint16_t iris) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(iris, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_IRIS_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the IRIS_RELATIVE control.
 * @param devh UVC device handle
 * @param[out] iris_rel TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_iris_rel(uvc_device_handle_t *devh, uint8_t* iris_rel, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_IRIS_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *iris_rel = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the IRIS_RELATIVE control.
 * @param devh UVC device handle
 * @param iris_rel TODO
 */
uvc_error_t uvc_set_iris_rel(uvc_device_handle_t *devh, uint8_t iris_rel) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = iris_rel;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_IRIS_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ZOOM_ABSOLUTE control.
 * @param devh UVC device handle
 * @param[out] focal_length TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_zoom_abs(uvc_device_handle_t *devh, uint16_t* focal_length, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_ZOOM_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *focal_length = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ZOOM_ABSOLUTE control.
 * @param devh UVC device handle
 * @param focal_length TODO
 */
uvc_error_t uvc_set_zoom_abs(uvc_device_handle_t *devh, uint16_t focal_length) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(focal_length, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_ZOOM_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ZOOM_RELATIVE control.
 * @param devh UVC device handle
 * @param[out] zoom_rel TODO
 * @param[out] digital_zoom TODO
 * @param[out] speed TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_zoom_rel(uvc_device_handle_t *devh, int8_t* zoom_rel, uint8_t* digital_zoom, uint8_t* speed, enum uvc_req_code req_code) {
  uint8_t data[3];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_ZOOM_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *zoom_rel = data[0];
    *digital_zoom = data[1];
    *speed = data[2];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ZOOM_RELATIVE control.
 * @param devh UVC device handle
 * @param zoom_rel TODO
 * @param digital_zoom TODO
 * @param speed TODO
 */
uvc_error_t uvc_set_zoom_rel(uvc_device_handle_t *devh, int8_t zoom_rel, uint8_t digital_zoom, uint8_t speed) {
  uint8_t data[3];
  uvc_error_t ret;

  data[0] = zoom_rel;
  data[1] = digital_zoom;
  data[2] = speed;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_ZOOM_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the PANTILT_ABSOLUTE control.
 * @param devh UVC device handle
 * @param[out] pan TODO
 * @param[out] tilt TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_pantilt_abs(uvc_device_handle_t *devh, int32_t* pan, int32_t* tilt, enum uvc_req_code req_code) {
  uint8_t data[8];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_PANTILT_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *pan = DW_TO_INT(data + 0);
    *tilt = DW_TO_INT(data + 4);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the PANTILT_ABSOLUTE control.
 * @param devh UVC device handle
 * @param pan TODO
 * @param tilt TODO
 */
uvc_error_t uvc_set_pantilt_abs(uvc_device_handle_t *devh, int32_t pan, int32_t tilt) {
  uint8_t data[8];
  uvc_error_t ret;

  INT_TO_DW(pan, data + 0);
  INT_TO_DW(tilt, data + 4);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_PANTILT_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the PANTILT_RELATIVE control.
 * @param devh UVC device handle
 * @param[out] pan_rel TODO
 * @param[out] pan_speed TODO
 * @param[out] tilt_rel TODO
 * @param[out] tilt_speed TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_pantilt_rel(uvc_device_handle_t *devh, int8_t* pan_rel, uint8_t* pan_speed, int8_t* tilt_rel, uint8_t* tilt_speed, enum uvc_req_code req_code) {
  uint8_t data[4];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_PANTILT_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *pan_rel = data[0];
    *pan_speed = data[1];
    *tilt_rel = data[2];
    *tilt_speed = data[3];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the PANTILT_RELATIVE control.
 * @param devh UVC device handle
 * @param pan_rel TODO
 * @param pan_speed TODO
 * @param tilt_rel TODO
 * @param tilt_speed TODO
 */
uvc_error_t uvc_set_pantilt_rel(uvc_device_handle_t *devh, int8_t pan_rel, uint8_t pan_speed, int8_t tilt_rel, uint8_t tilt_speed) {
  uint8_t data[4];
  uvc_error_t ret;

  data[0] = pan_rel;
  data[1] = pan_speed;
  data[2] = tilt_rel;
  data[3] = tilt_speed;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_PANTILT_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ROLL_ABSOLUTE control.
 * @param devh UVC device handle
 * @param[out] roll TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_roll_abs(uvc_device_handle_t *devh, int16_t* roll, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_ROLL_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *roll = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ROLL_ABSOLUTE control.
 * @param devh UVC device handle
 * @param roll TODO
 */
uvc_error_t uvc_set_roll_abs(uvc_device_handle_t *devh, int16_t roll) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(roll, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_ROLL_ABSOLUTE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ROLL_RELATIVE control.
 * @param devh UVC device handle
 * @param[out] roll_rel TODO
 * @param[out] speed TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_roll_rel(uvc_device_handle_t *devh, int8_t* roll_rel, uint8_t* speed, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_ROLL_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *roll_rel = data[0];
    *speed = data[1];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ROLL_RELATIVE control.
 * @param devh UVC device handle
 * @param roll_rel TODO
 * @param speed TODO
 */
uvc_error_t uvc_set_roll_rel(uvc_device_handle_t *devh, int8_t roll_rel, uint8_t speed) {
  uint8_t data[2];
  uvc_error_t ret;

  data[0] = roll_rel;
  data[1] = speed;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_ROLL_RELATIVE_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the PRIVACY control.
 * @param devh UVC device handle
 * @param[out] privacy TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_privacy(uvc_device_handle_t *devh, uint8_t* privacy, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_PRIVACY_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *privacy = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the PRIVACY control.
 * @param devh UVC device handle
 * @param privacy TODO
 */
uvc_error_t uvc_set_privacy(uvc_device_handle_t *devh, uint8_t privacy) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = privacy;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_PRIVACY_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the DIGITAL_WINDOW control.
 * @param devh UVC device handle
 * @param[out] window_top TODO
 * @param[out] window_left TODO
 * @param[out] window_bottom TODO
 * @param[out] window_right TODO
 * @param[out] num_steps TODO
 * @param[out] num_steps_units TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_digital_window(uvc_device_handle_t *devh, uint16_t* window_top, uint16_t* window_left, uint16_t* window_bottom, uint16_t* window_right, uint16_t* num_steps, uint16_t* num_steps_units, enum uvc_req_code req_code) {
  uint8_t data[12];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_DIGITAL_WINDOW_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *window_top = SW_TO_SHORT(data + 0);
    *window_left = SW_TO_SHORT(data + 2);
    *window_bottom = SW_TO_SHORT(data + 4);
    *window_right = SW_TO_SHORT(data + 6);
    *num_steps = SW_TO_SHORT(data + 8);
    *num_steps_units = SW_TO_SHORT(data + 10);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the DIGITAL_WINDOW control.
 * @param devh UVC device handle
 * @param window_top TODO
 * @param window_left TODO
 * @param window_bottom TODO
 * @param window_right TODO
 * @param num_steps TODO
 * @param num_steps_units TODO
 */
uvc_error_t uvc_set_digital_window(uvc_device_handle_t *devh, uint16_t window_top, uint16_t window_left, uint16_t window_bottom, uint16_t window_right, uint16_t num_steps, uint16_t num_steps_units) {
  uint8_t data[12];
  uvc_error_t ret;

  SHORT_TO_SW(window_top, data + 0);
  SHORT_TO_SW(window_left, data + 2);
  SHORT_TO_SW(window_bottom, data + 4);
  SHORT_TO_SW(window_right, data + 6);
  SHORT_TO_SW(num_steps, data + 8);
  SHORT_TO_SW(num_steps_units, data + 10);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_DIGITAL_WINDOW_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the REGION_OF_INTEREST control.
 * @param devh UVC device handle
 * @param[out] roi_top TODO
 * @param[out] roi_left TODO
 * @param[out] roi_bottom TODO
 * @param[out] roi_right TODO
 * @param[out] auto_controls TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_digital_roi(uvc_device_handle_t *devh, uint16_t* roi_top, uint16_t* roi_left, uint16_t* roi_bottom, uint16_t* roi_right, uint16_t* auto_controls, enum uvc_req_code req_code) {
  uint8_t data[10];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_CT_REGION_OF_INTEREST_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *roi_top = SW_TO_SHORT(data + 0);
    *roi_left = SW_TO_SHORT(data + 2);
    *roi_bottom = SW_TO_SHORT(data + 4);
    *roi_right = SW_TO_SHORT(data + 6);
    *auto_controls = SW_TO_SHORT(data + 8);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the REGION_OF_INTEREST control.
 * @param devh UVC device handle
 * @param roi_top TODO
 * @param roi_left TODO
 * @param roi_bottom TODO
 * @param roi_right TODO
 * @param auto_controls TODO
 */
uvc_error_t uvc_set_digital_roi(uvc_device_handle_t *devh, uint16_t roi_top, uint16_t roi_left, uint16_t roi_bottom, uint16_t roi_right, uint16_t auto_controls) {
  uint8_t data[10];
  uvc_error_t ret;

  SHORT_TO_SW(roi_top, data + 0);
  SHORT_TO_SW(roi_left, data + 2);
  SHORT_TO_SW(roi_bottom, data + 4);
  SHORT_TO_SW(roi_right, data + 6);
  SHORT_TO_SW(auto_controls, data + 8);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_CT_REGION_OF_INTEREST_CONTROL << 8,
    uvc_get_camera_terminal(devh)->bTerminalID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the BACKLIGHT_COMPENSATION control.
 * @param devh UVC device handle
 * @param[out] backlight_compensation device-dependent backlight compensation mode; zero means backlight compensation is disabled
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_backlight_compensation(uvc_device_handle_t *devh, uint16_t* backlight_compensation, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_BACKLIGHT_COMPENSATION_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *backlight_compensation = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the BACKLIGHT_COMPENSATION control.
 * @param devh UVC device handle
 * @param backlight_compensation device-dependent backlight compensation mode; zero means backlight compensation is disabled
 */
uvc_error_t uvc_set_backlight_compensation(uvc_device_handle_t *devh, uint16_t backlight_compensation) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(backlight_compensation, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_BACKLIGHT_COMPENSATION_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the BRIGHTNESS control.
 * @param devh UVC device handle
 * @param[out] brightness TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_brightness(uvc_device_handle_t *devh, int16_t* brightness, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_BRIGHTNESS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *brightness = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the BRIGHTNESS control.
 * @param devh UVC device handle
 * @param brightness TODO
 */
uvc_error_t uvc_set_brightness(uvc_device_handle_t *devh, int16_t brightness) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(brightness, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_BRIGHTNESS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the CONTRAST control.
 * @param devh UVC device handle
 * @param[out] contrast TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_contrast(uvc_device_handle_t *devh, uint16_t* contrast, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_CONTRAST_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *contrast = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the CONTRAST control.
 * @param devh UVC device handle
 * @param contrast TODO
 */
uvc_error_t uvc_set_contrast(uvc_device_handle_t *devh, uint16_t contrast) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(contrast, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_CONTRAST_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the CONTRAST_AUTO control.
 * @param devh UVC device handle
 * @param[out] contrast_auto TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_contrast_auto(uvc_device_handle_t *devh, uint8_t* contrast_auto, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_CONTRAST_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *contrast_auto = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the CONTRAST_AUTO control.
 * @param devh UVC device handle
 * @param contrast_auto TODO
 */
uvc_error_t uvc_set_contrast_auto(uvc_device_handle_t *devh, uint8_t contrast_auto) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = contrast_auto;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_CONTRAST_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the GAIN control.
 * @param devh UVC device handle
 * @param[out] gain TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_gain(uvc_device_handle_t *devh, uint16_t* gain, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_GAIN_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *gain = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the GAIN control.
 * @param devh UVC device handle
 * @param gain TODO
 */
uvc_error_t uvc_set_gain(uvc_device_handle_t *devh, uint16_t gain) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(gain, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_GAIN_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the POWER_LINE_FREQUENCY control.
 * @param devh UVC device handle
 * @param[out] power_line_frequency TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_power_line_frequency(uvc_device_handle_t *devh, uint8_t* power_line_frequency, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_POWER_LINE_FREQUENCY_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *power_line_frequency = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the POWER_LINE_FREQUENCY control.
 * @param devh UVC device handle
 * @param power_line_frequency TODO
 */
uvc_error_t uvc_set_power_line_frequency(uvc_device_handle_t *devh, uint8_t power_line_frequency) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = power_line_frequency;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_POWER_LINE_FREQUENCY_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the HUE control.
 * @param devh UVC device handle
 * @param[out] hue TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_hue(uvc_device_handle_t *devh, int16_t* hue, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_HUE_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *hue = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the HUE control.
 * @param devh UVC device handle
 * @param hue TODO
 */
uvc_error_t uvc_set_hue(uvc_device_handle_t *devh, int16_t hue) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(hue, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_HUE_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the HUE_AUTO control.
 * @param devh UVC device handle
 * @param[out] hue_auto TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_hue_auto(uvc_device_handle_t *devh, uint8_t* hue_auto, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_HUE_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *hue_auto = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the HUE_AUTO control.
 * @param devh UVC device handle
 * @param hue_auto TODO
 */
uvc_error_t uvc_set_hue_auto(uvc_device_handle_t *devh, uint8_t hue_auto) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = hue_auto;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_HUE_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the SATURATION control.
 * @param devh UVC device handle
 * @param[out] saturation TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_saturation(uvc_device_handle_t *devh, uint16_t* saturation, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_SATURATION_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *saturation = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the SATURATION control.
 * @param devh UVC device handle
 * @param saturation TODO
 */
uvc_error_t uvc_set_saturation(uvc_device_handle_t *devh, uint16_t saturation) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(saturation, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_SATURATION_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the SHARPNESS control.
 * @param devh UVC device handle
 * @param[out] sharpness TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_sharpness(uvc_device_handle_t *devh, uint16_t* sharpness, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_SHARPNESS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *sharpness = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the SHARPNESS control.
 * @param devh UVC device handle
 * @param sharpness TODO
 */
uvc_error_t uvc_set_sharpness(uvc_device_handle_t *devh, uint16_t sharpness) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(sharpness, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_SHARPNESS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the GAMMA control.
 * @param devh UVC device handle
 * @param[out] gamma TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_gamma(uvc_device_handle_t *devh, uint16_t* gamma, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_GAMMA_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *gamma = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the GAMMA control.
 * @param devh UVC device handle
 * @param gamma TODO
 */
uvc_error_t uvc_set_gamma(uvc_device_handle_t *devh, uint16_t gamma) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(gamma, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_GAMMA_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the WHITE_BALANCE_TEMPERATURE control.
 * @param devh UVC device handle
 * @param[out] temperature TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_white_balance_temperature(uvc_device_handle_t *devh, uint16_t* temperature, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *temperature = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the WHITE_BALANCE_TEMPERATURE control.
 * @param devh UVC device handle
 * @param temperature TODO
 */
uvc_error_t uvc_set_white_balance_temperature(uvc_device_handle_t *devh, uint16_t temperature) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(temperature, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the WHITE_BALANCE_TEMPERATURE_AUTO control.
 * @param devh UVC device handle
 * @param[out] temperature_auto TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_white_balance_temperature_auto(uvc_device_handle_t *devh, uint8_t* temperature_auto, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *temperature_auto = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the WHITE_BALANCE_TEMPERATURE_AUTO control.
 * @param devh UVC device handle
 * @param temperature_auto TODO
 */
uvc_error_t uvc_set_white_balance_temperature_auto(uvc_device_handle_t *devh, uint8_t temperature_auto) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = temperature_auto;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the WHITE_BALANCE_COMPONENT control.
 * @param devh UVC device handle
 * @param[out] blue TODO
 * @param[out] red TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_white_balance_component(uvc_device_handle_t *devh, uint16_t* blue, uint16_t* red, enum uvc_req_code req_code) {
  uint8_t data[4];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *blue = SW_TO_SHORT(data + 0);
    *red = SW_TO_SHORT(data + 2);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the WHITE_BALANCE_COMPONENT control.
 * @param devh UVC device handle
 * @param blue TODO
 * @param red TODO
 */
uvc_error_t uvc_set_white_balance_component(uvc_device_handle_t *devh, uint16_t blue, uint16_t red) {
  uint8_t data[4];
  uvc_error_t ret;

  SHORT_TO_SW(blue, data + 0);
  SHORT_TO_SW(red, data + 2);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the WHITE_BALANCE_COMPONENT_AUTO control.
 * @param devh UVC device handle
 * @param[out] white_balance_component_auto TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_white_balance_component_auto(uvc_device_handle_t *devh, uint8_t* white_balance_component_auto, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *white_balance_component_auto = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the WHITE_BALANCE_COMPONENT_AUTO control.
 * @param devh UVC device handle
 * @param white_balance_component_auto TODO
 */
uvc_error_t uvc_set_white_balance_component_auto(uvc_device_handle_t *devh, uint8_t white_balance_component_auto) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = white_balance_component_auto;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the DIGITAL_MULTIPLIER control.
 * @param devh UVC device handle
 * @param[out] multiplier_step TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_digital_multiplier(uvc_device_handle_t *devh, uint16_t* multiplier_step, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_DIGITAL_MULTIPLIER_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *multiplier_step = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the DIGITAL_MULTIPLIER control.
 * @param devh UVC device handle
 * @param multiplier_step TODO
 */
uvc_error_t uvc_set_digital_multiplier(uvc_device_handle_t *devh, uint16_t multiplier_step) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(multiplier_step, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_DIGITAL_MULTIPLIER_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the DIGITAL_MULTIPLIER_LIMIT control.
 * @param devh UVC device handle
 * @param[out] multiplier_step TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_digital_multiplier_limit(uvc_device_handle_t *devh, uint16_t* multiplier_step, enum uvc_req_code req_code) {
  uint8_t data[2];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_DIGITAL_MULTIPLIER_LIMIT_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *multiplier_step = SW_TO_SHORT(data + 0);
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the DIGITAL_MULTIPLIER_LIMIT control.
 * @param devh UVC device handle
 * @param multiplier_step TODO
 */
uvc_error_t uvc_set_digital_multiplier_limit(uvc_device_handle_t *devh, uint16_t multiplier_step) {
  uint8_t data[2];
  uvc_error_t ret;

  SHORT_TO_SW(multiplier_step, data + 0);

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_DIGITAL_MULTIPLIER_LIMIT_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ANALOG_VIDEO_STANDARD control.
 * @param devh UVC device handle
 * @param[out] video_standard TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_analog_video_standard(uvc_device_handle_t *devh, uint8_t* video_standard, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_ANALOG_VIDEO_STANDARD_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *video_standard = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ANALOG_VIDEO_STANDARD control.
 * @param devh UVC device handle
 * @param video_standard TODO
 */
uvc_error_t uvc_set_analog_video_standard(uvc_device_handle_t *devh, uint8_t video_standard) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = video_standard;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_ANALOG_VIDEO_STANDARD_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the ANALOG_LOCK_STATUS control.
 * @param devh UVC device handle
 * @param[out] status TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_analog_video_lock_status(uvc_device_handle_t *devh, uint8_t* status, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_PU_ANALOG_LOCK_STATUS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *status = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the ANALOG_LOCK_STATUS control.
 * @param devh UVC device handle
 * @param status TODO
 */
uvc_error_t uvc_set_analog_video_lock_status(uvc_device_handle_t *devh, uint8_t status) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = status;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_PU_ANALOG_LOCK_STATUS_CONTROL << 8,
    uvc_get_processing_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

/** @ingroup ctrl
 * @brief Reads the INPUT_SELECT control.
 * @param devh UVC device handle
 * @param[out] selector TODO
 * @param req_code UVC_GET_* request to execute
 */
uvc_error_t uvc_get_input_select(uvc_device_handle_t *devh, uint8_t* selector, enum uvc_req_code req_code) {
  uint8_t data[1];
  uvc_error_t ret;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_GET, req_code,
    UVC_SU_INPUT_SELECT_CONTROL << 8,
    uvc_get_selector_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data)) {
    *selector = data[0];
    return UVC_SUCCESS;
  } else {
    return ret;
  }
}


/** @ingroup ctrl
 * @brief Sets the INPUT_SELECT control.
 * @param devh UVC device handle
 * @param selector TODO
 */
uvc_error_t uvc_set_input_select(uvc_device_handle_t *devh, uint8_t selector) {
  uint8_t data[1];
  uvc_error_t ret;

  data[0] = selector;

  ret = (uvc_error_t)libusb_control_transfer(
    devh->usb_devh,
    REQ_TYPE_SET, UVC_SET_CUR,
    UVC_SU_INPUT_SELECT_CONTROL << 8,
    uvc_get_selector_units(devh)->bUnitID << 8 | devh->info->ctrl_if.bInterfaceNumber,
    data,
    sizeof(data),
    0);

  if (ret == sizeof(data))
    return UVC_SUCCESS;
  else
    return ret;
}

#endif // CTRL_GEN_C