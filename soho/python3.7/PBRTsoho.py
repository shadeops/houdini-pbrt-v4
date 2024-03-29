import soho

from PBRTnodes import PBRTParam


class SohoPBRT(soho.SohoParm):
    """Simple subclass of soho.SohoParm that adds to_pbrt() method"""

    def to_pbrt(self, pbrt_type=None):
        """Convert SohoParm to PBRTParam"""
        # bounds not supported
        # shader not supported
        if pbrt_type is None:
            to_pbrt_type = {"real": "float", "fpreal": "float", "int": "integer"}
            pbrt_type = to_pbrt_type.get(self.Type, self.Type)
        pbrt_name = self.Key
        return PBRTParam(pbrt_type, pbrt_name, self.Value)
