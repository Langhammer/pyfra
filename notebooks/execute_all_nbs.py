from pathlib import Path
import papermill

for nb in Path('.').glob('pyfra_nb_*.ipynb'):
    papermill.execute_notebook(
        input_path=nb,
        output_path=nb,
        kernel_name='Python3'
    )