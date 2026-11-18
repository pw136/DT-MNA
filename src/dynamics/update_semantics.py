
from enum import Enum

class ExposureSemantics(str, Enum):

    GROUPED_ADDITIVE = "grouped_additive"
    INDEPENDENT_EXPOSURES = "independent_exposures"

class JointUpdateSemantics(str, Enum):

    INDEPENDENT_NODES = "independent_nodes"
    COMMON_SHOCK = "common_shock"

def coerce_exposure_semantics(value):
    if isinstance(value, ExposureSemantics):
        return value
    return ExposureSemantics(value)

def coerce_joint_semantics(value):
    if isinstance(value, JointUpdateSemantics):
        return value
    return JointUpdateSemantics(value)

