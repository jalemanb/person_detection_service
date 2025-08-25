import cv2
import numpy as np
from cv_bridge.boost.cv_bridge_boost import cvtColor2


def compressed_imgmsg_to_cv2(cmprs_img_msg, desired_encoding='passthrough'):
    """
    Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.

    :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
    :param desired_encoding:  The encoding of the image data, one of the following strings:

        * ``"passthrough"``
        * one of the standard strings in sensor_msgs/image_encodings.h

    :rtype: :cpp:type:`cv::Mat`
    :raises CvBridgeError: when conversion is not possible.

    If desired_encoding is ``"passthrough"``, then the returned image has the same format
    as img_msg. Otherwise desired_encoding must be one of the standard image encodings

    This function returns an OpenCV :cpp:type:`cv::Mat` message on success,
    or raises :exc:`cv_bridge.CvBridgeError` on failure.

    If the image only has one channel, the shape has size 2 (width and height)
    """

    str_msg = cmprs_img_msg.data
    img_encoding = cmprs_img_msg.format.split(';')[0]

    buf = np.ndarray(shape=(1, len(str_msg)),
                        dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if desired_encoding == 'passthrough':
        return im

    try:
        res = cvtColor2(im, img_encoding, desired_encoding)
    except RuntimeError as e:
        raise e

    return res