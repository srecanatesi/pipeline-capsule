# Local copy for capsule: import and run via main
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from hmm_crossvalidation_fast import *  # noqa: F401,F403

if __name__ == '__main__':
    main()




