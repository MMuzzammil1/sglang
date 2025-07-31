from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    LOOKAHEAD = auto()
    EAGLE3 = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE
    
    def is_none_or_lookahead(self):
        return self == SpeculativeAlgorithm.NONE or self == SpeculativeAlgorithm.LOOKAHEAD

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_lookahead(self):
        return self == SpeculativeAlgorithm.LOOKAHEAD

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "LOOKAHEAD": SpeculativeAlgorithm.LOOKAHEAD,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
