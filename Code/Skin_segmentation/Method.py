from enum import Enum, auto
class Method(Enum):
	'''Available methods for processing an image
			REGION_BASED: segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm'''
	REGION_BASED = auto()