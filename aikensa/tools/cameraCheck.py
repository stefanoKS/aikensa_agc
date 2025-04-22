import time
import sys
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst

def main():

    Gst.init(sys.argv)  # init gstreamer

    serial = None

    pipeline = Gst.parse_launch("tcambin name=bin "
                                " ! videoconvert"
                                " ! ximagesink sync=false")

    # retrieve the bin element from the pipeline
    camera = pipeline.get_by_name("bin")

    # serial is defined, thus make the source open that device
    if serial is not None:
        camera.set_property("serial", serial)

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

if __name__ == "__main__":
    main()