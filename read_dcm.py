import pydicom
import numpy
img = pydicom.dcmread("data/images/1.2.246.512.1.2.0.1.6004323248482.184711860.20230104090221-2-49663-1952zqu.dcm")
# reader = gdcm.ImageReader()
# reader.SetFileName("./data/images/1.2.246.512.1.2.0.1.6004323248482.184711860.20230104090221-2-49663-1952zqu.dcm")
# ret = reader.Read()
# for itm in img.keys().__iter__():
#     print(itm[1])
print(img.keys().__len__())
print(img[(0x0008, 0x0018)])
print(img.SOPInstanceUID)
print(img[0x20,0x10])
# print(numpy.frombuffer(img.values().))
# print(img.PixelData)
print(img.pixel_array)
# img = reader.Get

# print(type(reader))
