# Development Manual

## Technical Architecture
The following are the requirements to run the client-server system:

- **Operating System:** Linux or Windows Subsystem for Linux 2
- **Storage Space** 4.5GB free disk space

Linux is required to run the system. If using Windows, a Linux environment can be ran using the Windows Susbsytem for Linux. At least 4.5GB free disk space is needed considering the packages and dependencies used to run the system. Internet connection is also required since the client downloads required files, the **features_and_classes.txt** file and the **client.zip** file, upon launching of the client GUI application.

## Installation of Required Tools
The following tools are required before setting up the client-server system on your local machine:

### `Python 3.10.x`
Python is the main language used in developing the system. 

If using Windows:
You may download the installer available at https://www.python.org/downloads/ and follow the installation guide on https://www.digitalocean.com/community/tutorials/install-python-windows-10.

If using Ubuntu (Linux):
Python is already pre-installed in Ubuntu 22.04. You can simply check the version if it is 3.10. If not, you may follow the guide on https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux-system/ for installation.

### `Git`
This will be used to access the files stored in the system's GitHub repository. 

If using Windows:
You may download the installer available at https://git-scm.com/downloads and follow the installation guide on https://www.simplilearn.com/tutorials/git-tutorial/git-installation-on-windows#git_installation_on_windows.

If using Ubuntu (Linux):
Follow the instructions available at https://www.digitalocean.com/community/tutorials/how-to-install-git-on-ubuntu-20-04 to install Git in Ubuntu.

### `Windows Subsystem for Linux 2 (WSL2)` (only required for those using Windows)
This will allow running of ConcreteML which is the main library used to implement FHE inference. Follow the installation guide available at https://learn.microsoft.com/en-us/windows/wsl/install. Make sure to change the default Linux distribution to Ubuntu-22.04.

## System Set Up
1. Make sure that GUI apps / GUI packages are supported in your system.
   - If using Windows:
     - To make sure that your WSL2 includes Linux GUI support, follow the **Existing WSL Install** section of the guide in https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps.
     - After the previous step, you may open your WSL2 terminal again. Then, run the following commands:
       ``` {.bash language="bash"}
       sudo apt install x11-apps -y
       ```
       ``` {.bash language="bash"}
       export DISPLAY=:0;
       ```
   - If using Ubuntu (Linux):
     - Run the following command to make sure that packages for GUI (i.e. tkinter) will work:
       ``` {.bash language="bash"}
       sudo apt install python3.10-tk
       ```
2. After installing the above required tools, you may proceed with cloning the project GitHub repository at https://github.com/scg-upm/sp-concreteML-BrainTumorClassifier.git using the following command:
   ``` {.bash language="bash"}
   git clone https://github.com/scg-upm/sp-concreteML-BrainTumorClassifier.git
   ```
4. Download the required packages for running the client-server system.
   - If using Windows:
     - In Windows File Explorer, open the folder of the cloned GitHub repository. Double click on the **src** folder to go to the source code directory. Open the WSL2 terminal by typing **bash** on the address bar then press **Enter**. Doing this will open the WSL2 terminal and automatically change the directory to your **src** folder. 
   - If using Linux:
     - In Files, open the folder of the cloned GitHub repository. Right click on the **src** folder and select **Open in terminal**. This will open the terminal and automatically set the directory to the **src** folder.
   - Once you have opened your terminal in the source code directory, run the following command to install the required packages:
     ``` {.bash language="bash"}
     pip install -r requirements.txt
     ```
5. Once the required packages are successfully installed, you can now run the client-server system. Refer to the [User Manual](HELP.md) for instructions on running the system.
