
## Python Dense Optical Flow

**Python** wrapper for Ce Liu's [C++ implementation](https://people.csail.mit.edu/celiu/OpticalFlow/) of Coarse2Fine Optical Flow. This is **super fast and accurate** optical flow method based on Coarse2Fine warping method from Thomas Brox. This python wrapper has minimal dependencies, and it also eliminates the need for C++ OpenCV library. For real time performance, one can additionally resize the images to a smaller size.

Run the following steps to download, install and demo the library:
  ```Shell
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

This wrapper code was developed as part of our [CVPR 2017 paper on Unsupervised Learning using unlabeled videos](http://cs.berkeley.edu/~pathak/unsupervised_video/). Github repository for our CVPR 17 paper is [here](https://github.com/pathak22/unsupervised-video).


## Fork Modification - Verbose Argument

Added a `verbose` argument to the `coarse2fine_flow` function that allows the user to enable/disable any non-error related output from the function. It defaults to **False**.

Setup and usage of the library is identical to the original wrapper.

The `demo.py` file has been updated make use of the new `verbose` argument for the sake of example. Besides the inclusion of the new extra argument the behavior is identical to the original `demo.py` file.


