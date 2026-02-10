
# TODO: Move program parsing variables to here from simulator config

# OUTPUTLayers
StartAddressSize = 32
NumberOfBlocksSize = 16
ParamSize = 16
BatchSize = 16
NumOfOutputChannelsSize = 16
HeightSize = 16
WidthSize = 16
ZPSize = 8
ReserveSize = 8
XFoldSize = 8
YFoldSize = 8
ScaleSize = 32
NameSize = 64

StartAddressOffest = 0
NumberOfBlocksOffset = StartAddressOffest + StartAddressSize
ParamOffset = NumberOfBlocksOffset + NumberOfBlocksSize
BatchOffset = ParamOffset + ParamSize
NumOfOutputChannelsOffset = BatchOffset + BatchSize
HeightOffset = NumOfOutputChannelsOffset+ NumOfOutputChannelsSize
WidthOffset = HeightOffset + HeightSize
ZPOffset = WidthOffset + WidthSize
ReserveOffset = ZPOffset + ZPSize
YFoldOffset = ReserveOffset + ReserveSize
XFoldOffset = YFoldOffset + YFoldSize
ScaleOffset = XFoldOffset + XFoldSize 
NameOffset = ScaleOffset + ScaleSize

EndOffest = StartAddressSize + NumberOfBlocksSize + ParamSize + BatchSize + NumOfOutputChannelsSize + HeightSize + WidthSize + ZPSize + ReserveSize + ScaleSize + NameSize