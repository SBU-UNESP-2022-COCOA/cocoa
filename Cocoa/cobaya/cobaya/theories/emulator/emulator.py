### Note (KZ, Feb22/2023): this theory code now is designed for now just to avoid calling camb,
### and do some simple change of variable. The actually emulator usage is in likelihood/
### One could add the emulator here for code simplicity but some specifications for emulator
### might not be very straight forward.

from cobaya.theory import Theory

class Emulator(Theory):

    def initialize(self):
        self.log.info("Initialized!")
    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        return {'H0': None}

    def must_provide(self, **requirements):
        if 'H0' in requirements:
            return {}

    def get_can_provide_params(self):
        return ['h']

    def calculate(self, state, want_derived=True, **params_values_dict):
        state['H0'] = self.provider.get_param('H0')
        state['derived'] = {'h': 0.7}

    def get_H0(self):
        return self.current_state['H0']