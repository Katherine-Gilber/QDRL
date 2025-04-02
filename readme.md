# Environment settings

python3.7.5

pip install -r requirements.txt

# The corresponding file modification in changefiles
| Filename       | Local position                                                           |Modification method|
|-----------------|------------------------------------------------------------------------------|----|
| torch.py | .\Anaconda3\envs\py37\Lib\site-packages\pennylane\qnn\torch.py |Copy the file to overwrite the local file with the same name.|
| graphenv.py     | .\Anaconda3\envs\py37\Lib\site-packages\gym\envs\classic_control\myenv      |Create a new folder named ”myenv“ under the "classic_control" folder, and copy the file to the "classic_control/myenv" folder.|
| classic_init.py | .\Anaconda3\envs\py37\Lib\site-packages\gym\envs\classic_control\\_init_.py | Copy the contents of the file to the end of "init.py".       |
| env_init.py     | .\Anaconda3\envs\py37\Lib\site-packages\gym\envs\\_init_.py                 |Copy the contents of the file to the end of "init.py".|

