# arrow-finder
A Python module that finds arrow from Raspberry Pi camara

## Usage
For example, using this module from a script `main.py` under the folder containing this module:
```
src/
    main.py
    ArrowFinder
```
```python
import cv2
from ArrowFinder import ArrowFinder

# Create ArrowFinder object
af = ArrowFinder()

# Using getArrows method to capture image and detect arrows
# The withImage argument defaults to False, in which case the image 
# with bounding box and label won't be generated and returned.
result = af.getArrows(withImage=True)

# Obtaining the arrows detected
arrows = result['arrows']
print(arrows) # [{'box': (919, 595, 277, 279), 'dir': 'ARROW_LEFT'}]

# Displaying the image with bounding box and label
im = result['image']
cv2.imshow("Img", im)
```

## Dependencies & Setup

This module makes use of CV related algorithms in [OpenCV](https://github.com/opencv/opencv).

To setup `OpenCV`, run the following command:
```
sudo bash setup.sh
```
when the text editor pops up for the first time, change the `CONF_SWAPSIZE` line to be `CONF_SWAPSIZE=1024`. When it pops up for the second time, change the same line to the original value.

Python library used in this module can be found in `requirements.txt`, they can be installed with
```
pip install -r requirements.txt
```
