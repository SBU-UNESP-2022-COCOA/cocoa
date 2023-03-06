from .baseconfig import F2003Class, fortran_class, numpy_1d, CAMBError, np, \
    AllocatableArrayDouble, f_pointer
from ctypes import c_int, c_double, byref, POINTER, c_bool

@fortran_class
class DarkEnergyModel(F2003Class):
    
    _fortran_class_module_ = 'DarkEnergyInterface'
    _fortran_class_name_ = 'TDarkEnergyModel'
    
    _fields_ = [
        ("__is_cosmological_constant", c_bool),
        ("__num_perturb_equations", c_int),
        ("w0", c_double),
        ("w1", c_double),
        ("w2", c_double),
        ("w3", c_double),
        ("w4", c_double),
        ("w5", c_double),
        ("a1", c_double),
        ("a2", c_double),
        ("a3", c_double),
        ("a_min", c_double),
        ("state", c_int),
        ("__no_perturbations", c_bool, "turn off perturbations (unphysical, so hidden in Python)")
    ]
                
#    def validate_params(self):
#        if self.wa + self.w > 0:
#            raise CAMBError('dark energy model has w + wa > 0, giving w>0 at high redshift')
#        return True

    def set_params(self, w0=-1.0, w1=0, w2=0, w3=0, w4=0, w5=0, a1=0.25, a2=0.5, a3=0.75, a_min=0.0005, state=0):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a_min = a_min
        self.state = state
#        self.validate_params()

# short names for models that support w/wa
F2003Class._class_names.update({'ppf': DarkEnergyModel})
