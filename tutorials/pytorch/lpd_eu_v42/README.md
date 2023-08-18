# Model Card: lpd_eu_v42

## Description:
This is an object detection model for license plates. The model design is based on yolov5n. Additional output parameters were added to accomodate license plates that are rotated or skewed. Accomodating for skew helps a recognition model correctly decode the characters on the plate. This model was trained on plates from the European Union (EU). It may not work correctly for plates from other regions.

## Input:
(1x3x288x1024), BGR colors, 288 height, 1024 width

The input is an image. The input shape is intended to process the bottom half of a standard 16x9 frame.

## Output:
(1x6x36x128) Output_Str8_Shape  
(1x1x36x128) Output_Str8_Obj  
(1x6x18x64) Output_Str16_Shape  
(1x1x18x64) Output_Str16_Obj  
(1x6x9x32) Output_Str32_Shape  
(1x1x9x32) Output_Str32_Obj

There are two output each for strides 8, 16, and 32. Each stride has a different anchor shape. The "object" output contains the object detection score at each location. The "shape" output contains the following paramenters for each location:
```
[center x, center y, width, height, skew x, skew y]
```
These outputs must go through the post-processing routine to be converted to units of pixels. See the post-processing code for details.