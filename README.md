# sp-concreteML-WDTCRecurrenceClassifier

## System Description
The client-server system presented here is a fully homomorphic encryption-based logistic regression model for multi-class tumor classification that uses genomic data as its input. It allows clients to preprocess and encrypt their gene expression data then send them to the server for FHE inference. The system was developed on Windows Subsystem for Linux 2 (WSL2). The client-side consists of a graphical user interface (GUI) mainly built with Tkinter, CustomTkinter. The server-side is a web application built with Django web framework. Both sides of the system are also built using ConcreteML library for the implementation of main functionalities: key generation, encryption, decryption, and FHE inference.

## Installation and Set Up
Users may set up the system in two ways:
1. Manual Set Up
   - Please refer to the [Development Manual](https://github.com/scg-upm/sp-concreteML-BrainTumorClassifier/blob/main/manual/DEVELOPMENT.md) for instructions on setting up the system from scratch.
2. Virtual Machine (.ova)
   - If users want to use a pre-configured version of the system to simply see the app and test its functionalities, they may access the virtual machine .ova file at https://drive.google.com/file/d/1qBe6diH2h58A3M7RjVaipYQpkrGZCyxR/view?usp=drive_link and import this in a virtualization software such as Virtualbox.

**Note:** If using a virtual machine on Windows, it is required to disable Hyper-V on the host machine. You can do this by following the instructions at https://www.makeuseof.com/windows-11-disable-hyper-v/.

Once done with the system set-up, users can refer to the [User Manual](https://github.com/scg-upm/sp-concreteML-BrainTumorClassifier/blob/main/manual/HELP.md) for instructions on running the system. 
