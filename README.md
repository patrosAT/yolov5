# YOLO v5 - Class Detector

This repository is a fork of [Ultralytics' YOLO v5 repo](https://github.com/ultralytics/yolov5) that has been adapted to fit a specific in the domain of [human-to-robot handovers](https://patrosat.github.io/h2r_handovers/): This repo can be used to suplement the [YOLO v3 module](https://github.com/leggedrobotics/darknet_ros).

**All code and models are under active development, and are subject to modification or deletion without notice. Use at your own risk.**


## Requirements

* Python 3.8
* Torch >= 1.6

To install run:
```bash
$ pip3 install -r requirements.txt
```

## Usage

### General Usage

First, import the YOLOv5Detector class from detector.py.

```bash
from detector import YOLOv5Detector
```

Then, create a detecor instance.

```bash
detector = YOLOv5Detector()
```

Finally, use the detect() function to pass an **BGR**-image (type: [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)) and receive the result.

```bash
res = detector.detect(image)
```

The function returns a [<class 'numpy.ndarray'>](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) containing the bounding box coordinates, confidence, and class label of all findings:
[[x<sub>0<sub>00</sub></sub> y<sub>0<sub>00</sub></sub> x<sub>0<sub>11</sub></sub> y<sub>0<sub>11</sub></sub> conf<sub>0</sub> class<sub>0</sub>] 
[x<sub>1<sub>00</sub></sub> y<sub>1<sub>00</sub></sub> x<sub>1<sub>11</sub></sub> y<sub>1<sub>11</sub></sub> conf<sub>1</sub> class<sub>1</sub>] 
... ]

```bash
print(res[:4])

Out:
[[        239         645         312         688     0.83059           41]
 [        972         752        1053         816      0.8106           41]
 [         29         614         102         656     0.76942           41]
 [        707         633         766         688     0.74638           41]]
 ```

### Classes

The class names are stored in the class variable 'names'.

```bash
print(detector.names)

Out:
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

Additionally, a unique color code for each class is stored in the 'colors' class variable.

```bash
print(detector.colors)

Out:
[[148, 205, 176], [120, 63, 222], [213, 185, 28], [103, 93, 218], [254, 130, 216], [128, 56, 204], [24, 191, 182], [32, 153, 89], [131, 67, 139], [109, 16, 13], [245, 194, 16], [119, 247, 232], [28, 122, 248], [13, 254, 151], [145, 221, 131], [205, 31, 61], [120, 224, 12], [249, 207, 114], [20, 208, 148], [173, 2, 152], [49, 73, 96], [136, 52, 208], [65, 26, 165], [74, 123, 169], [72, 235, 64], [185, 46, 69], [235, 230, 35], [38, 162, 16], [194, 1, 155], [39, 35, 41], [177, 160, 133], [29, 11, 243], [66, 124, 188], [254, 196, 169], [30, 249, 51], [151, 80, 157], [104, 236, 102], [159, 202, 14], [162, 189, 168], [229, 123, 71], [166, 39, 122], [34, 24, 110], [63, 180, 115], [82, 13, 56], [11, 42, 185], [170, 112, 134], [168, 137, 214], [243, 182, 166], [107, 4, 16], [69, 1, 221], [51, 138, 225], [82, 57, 192], [94, 89, 128], [118, 205, 38], [243, 88, 207], [120, 50, 107], [59, 86, 173], [8, 238, 61], [164, 46, 34], [223, 36, 244], [100, 15, 167], [197, 203, 131], [85, 11, 204], [29, 85, 21], [33, 82, 46], [3, 48, 12], [158, 199, 207], [20, 167, 210], [217, 254, 73], [84, 154, 228], [235, 204, 220], [108, 120, 32], [49, 195, 236], [247, 249, 150], [107, 16, 219], [74, 240, 70], [236, 87, 72], [178, 104, 9], [24, 137, 117], [115, 132, 83]]

```


## References

### YOLO v5

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

### Human-to-Robot Handovers 

The paper can be found on [arXiv](https://arxiv.org/abs/2006.01797), the code can be foung on [github](https://patrosat.github.io/h2r_handovers/).

## Disclaimer

Please keep in mind that no system is 100% fault tolerant and that this demonstrator is focused on pushing the boundaries of innovation. Careless interaction with robots can lead to serious injuries, always use appropriate caution!

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.