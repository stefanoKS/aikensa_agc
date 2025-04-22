#!/usr/bin/env python3

#
# This example will show you how to start a simply live stream
#

import time
import sys
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst


def main():
    """
    """
    Gst.init(sys.argv)  # init gstreamer

    # this line sets the gstreamer default logging level
    # it can be removed in normal applications
    # gstreamer logging can contain verry useful information
    # when debugging your application
    # see https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html
    # for further details
    Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)

    serial = "02420129"

    pipeline = Gst.parse_launch("tcambin name=bin"
                                " ! capsfilter name=filter"
                                " ! videoconvert"
                                " ! ximagesink")

    # serial is defined, thus make the source open that device
    if serial:
        # retrieve the bin element from the pipeline
        camera = pipeline.get_by_name("bin")
        camera.set_property("serial", serial)

    caps = Gst.Caps.new_empty()

    structure = Gst.Structure.new_from_string("video/x-raw")
    structure.set_value("width", 640)
    structure.set_value("height", 480)

    try:
        fraction = Gst.Fraction(30, 1)
        structure.set_value("framerate", fraction)
    except TypeError:
        struc_string = structure.to_string()

        struc_string += ",framerate={}/{}".format(30, 1)
        structure.free()
        structure, end = structure.from_string(struc_string)

    caps.append_structure(structure)

    structure.free()
    # structure is not useable from here on

    capsfilter = pipeline.get_by_name("filter")

    if not capsfilter:
        print("Could not retrieve capsfilter from pipeline.")
        return 1

    capsfilter.set_property("caps", caps)

    # to statically create caps you can reduce the whole procedure to
    # caps = Gst.Caps.from_string("video/x-raw,format=BGRx,width=640,height=480,framerate=30/1");
    #
    # Gst.parse_lauch allows you to use strings for caps descriptions.
    # That means everything until now can be done with:
    #
    # pipeline = Gst.parse_launch("tcambin name=source"
    #                             " ! video/x-raw,format=BGRx,width=640,height=480,framerate=30/1"
    #                             " ! videoconvert ! ximagesink");

    pipeline.set_state(Gst.State.PLAYING)

    print("Press Ctrl-C to stop.")

    # We wait with this thread until a
    # KeyboardInterrupt in the form of a Ctrl-C
    # arrives. This will cause the pipline
    # to be set to state NULL
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

    return 0


if __name__ == "__main__":
    sys.exit(main())