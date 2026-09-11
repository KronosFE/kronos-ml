"""Import every member module so it self-registers into the fleet.
Each module imports only numpy + base at load time (torch/sklearn are lazy), so
this stays import-clean in a bare environment."""
from . import spine        # noqa: F401  KHALO, KGATE
from . import transport    # noqa: F401  KYRO
from . import state        # noqa: F401  KFLOW, KEYE
from . import magnet       # noqa: F401  KOIL
from . import materials    # noqa: F401  KMAT
from . import disrupt      # noqa: F401  KWARD
from . import equil        # noqa: F401  KORE
from . import control      # noqa: F401  KAIROS
from . import plant        # noqa: F401  KBURN, KPATH, KISO
from . import quantum      # noqa: F401  KQUBIT, KQROSS, KSENSE, KTENSOR
from . import fusion       # noqa: F401  KFUSE
from . import advanced     # noqa: F401  KLAW, KDRIVE, KGEN, KFUEL
from . import capstone     # noqa: F401  KFORGE, KPILOT
from . import neutronics   # noqa: F401  KBREED, KFLUX
from . import roadmap      # noqa: F401  KHEAT, KEDGE, KRAD, KECON
