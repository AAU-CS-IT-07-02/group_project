## Project SetUp SINDy.
Materials:
- https://pysindy.readthedocs.io/en/latest/
- https://www.youtube.com/watch?v=DvbbXX8Bd90

Setup: In the terminal of your IDE of choice use the command "pip install pysindy"
There is a sample code in the website.
The video explains a little of what SINDy does.

## Python venv SetUp
Materials:
- https://medium.com/@royce963/setting-up-jupyterlab-and-a-virtual-environment-c79002e0e5f7
- https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
- https://www.w3schools.com/python/python_virtualenv.asp

The steps for creating a venv:
- change the directory to the directory of your project
- run ```python3 -m venv venv```
- run ```.\venv\Scripts\Activate.ps1``` for Windows PowerShell
- run `> cd venv/Scripts` and `> activate` for command prompt  
- run ```source venv/bin/activate``` for linux or macOS
- install the requirements using the command ```pip install -r requirements.txt```
- you can use whatever IDE you want, I have attached links for JetBrains PyCharm and JupyterLab
- when you are done using the venv run the command "deactivate" in the Windows PowerShell
