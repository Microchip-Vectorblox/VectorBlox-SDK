# Model Card: lpr_eu_v3

## Description:
This is a character recognition model for license plates. This model was trained on plates from the European Union (EU). It may not work correctly for plates from other regions.

## Input:
(1x3x34x146), BGR colors, 34 height, 146 width

The input is an image of a license plate. For best performance the image should be warped and cropped so that the edges of the license plate are at the edges of the input image.

## Output:
(1x37x1x18)

The output is a 2d array. The size 18 dimension represents the possible characters, from left to right. When decoding the output, repepated characters should be removed. The size 37 dimension represents the scores for each possible character from the set below. The last element in the set represents no character.
 
```
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
 'W', 'X', 'Y', 'Z', 'I', 'O', '-']
```
